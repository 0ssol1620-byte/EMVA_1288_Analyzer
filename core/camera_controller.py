#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import logging, threading, time, subprocess, os, json
from typing import Optional, List, Dict, Any, Tuple, Literal
import numpy as np
from contextlib import contextmanager
from datetime import datetime
from contextlib import suppress
from . import controller_pool

# ──────────────────────────────────────────────
#  1) eGrabber (실제 / 더미)  +  싱글턴
# ──────────────────────────────────────────────
try:
    from egrabber import (
        EGenTL, EGrabber, Buffer,
        BUFFER_INFO_BASE, BUFFER_INFO_WIDTH,
        BUFFER_INFO_DELIVERED_IMAGEHEIGHT, BUFFER_INFO_DATA_SIZE,
        INFO_DATATYPE_PTR, INFO_DATATYPE_SIZET,
        TimeoutException, GenTLException,
        query, Coaxlink
    )
    EURESYS_AVAILABLE = True

    # ──────────────────────────────────────────────
    #  글로벌 EGenTL 싱글턴 (안정판)
    # ──────────────────────────────────────────────
    _gentl_singleton: Optional[EGenTL] = None

    def _new_gentl() -> EGenTL:
        """CTI 경로가 깨져도 최소한 객체는 리턴하도록 2단 시도."""
        try:
            return EGenTL(Coaxlink())  # 표준
        except Exception:
            return EGenTL()  # fallback


    def _gentl_is_healthy(g) -> bool:
        """
        빌드마다 존재하는 '가벼운' API 한 가지라도 호출에 성공하면 정상으로 본다.
        호출 자체가 없는 경우도 *정상* 취급 (AttributeError 허용).
        """
        try:
            for probe in (
                    lambda: getattr(g, "get_num_interfaces")() if hasattr(g, "get_num_interfaces") else None,
                    lambda: len(g.interfaces()) if callable(getattr(g, "interfaces", None)) else None,
                    lambda: len(g.interfaces) if hasattr(g, "interfaces") else None,
                    lambda: getattr(g, "interface_count")() if callable(getattr(g, "interface_count", None)) else None,
                    lambda: getattr(g, "version")() if callable(getattr(g, "version", None)) else None,
            ):
                try:
                    probe();
                    return True
                except AttributeError:
                    continue  # 메서드가 없으면 다른 probe 시도
            return True  # 모든 probe 가 AttributeError → 기능은 없지만 죽진 않음
        except GenTLException as e:
            return e.gc_err == 0  # 0==SUCCESS → 정상, 그 외 == 오류
        except Exception:
            return False


    def _get_gentl() -> EGenTL:
        """깨졌으면 폐기 후 새로 만든다."""
        global _gentl_singleton
        if _gentl_singleton is None or not _gentl_is_healthy(_gentl_singleton):
            try:
                # 가능하면 기존 객체 정리 (3.x 에서만 close() 존재)
                getattr(_gentl_singleton, "close", lambda: None)()
            except Exception:
                pass
            _gentl_singleton = _new_gentl()
        return _gentl_singleton


except ImportError:  # ───── 시뮬레이션 모드 ─────
    EURESYS_AVAILABLE = False

    class EGenTL:                         # 더미
        def get_num_interfaces(self): return 0
        def get_interface_id(self, idx): raise IndexError
        @contextmanager
        def open_interface(self, iface_id): yield DummyInterface()

    class EGrabber:  pass
    class Buffer:    ...                  # 생략 (더미)
    class TimeoutException(Exception): pass
    class GenTLException(Exception):   pass
    query = None

    _get_gentl = lambda: EGenTL()         # 중복 생성 OK

class DummyInterface:
    def get_num_devices(self): return 0
    def get_device_id(self, idx): raise IndexError

from core.camera_exceptions import (
    CameraError, CameraConnectionError, CameraNotConnectedError, GrabberError,
    GrabberNotActiveError, FrameAcquisitionError, FrameTimeoutError,
    GrabberStartError, GrabberStopError, ParameterError, ParameterSetError,
    ParameterGetError, CommandExecutionError
)

logger = logging.getLogger(__name__)

from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer
LEGACY_MAP: Dict[str, List[Tuple[str, Any]]] = {}
# ────────────────────────────────────────────────────────────────────
#  Memento CLI 경로 탐색
# ────────────────────────────────────────────────────────────────────
def _find_memento_cli() -> Optional[str]:
    exe = "memento.exe" if os.name == "nt" else "memento"
    env = os.environ.get("EURESYS_MEMENTO_PATH")
    if env and os.path.isfile(env):
        return env
    for p in os.environ.get("PATH", "").split(os.pathsep):
        f = os.path.join(p, exe)
        if os.path.isfile(f):
            return f
    for f in [
        r"C:\Program Files\Euresys\Memento\bin\x86_64\memento.exe",
        r"C:\Program Files\Euresys\Memento\bin\memento.exe",
        r"C:\Program Files\Euresys\eGrabber\bin\memento.exe",
    ]:
        if os.path.isfile(f):
            return f
    return None

def _run_memento_cmd(args: List[str]) -> None:
    cli = _find_memento_cli()
    if not cli:
        raise FileNotFoundError("Memento CLI not found")
    subprocess.check_call([cli, *args],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
# ──────────────────────────────────────────────────────
#  (1) 디버그용 – 실제 EGenTL 객체가 어떤 API를 갖고 있는지 확인
# ──────────────────────────────────────────────────────
def debug_print_gentl_dir():
    """
    현재 설치된 eGrabber Python-API 가 노출하는 속성·메서드를 한눈에 출력.
    - main 스크립트나 IPython에서 임시로 호출해 보세요.
    """
    try:
        g = _get_gentl()
        print("EGenTL dir =", dir(g))
    except Exception as e:
        print("EGenTL dir() 확인 실패:", e)
# ──────────────────────────────────────────────────────
#  (2) 인터페이스 열거 – API 버전별로 안전하게 대응
# ──────────────────────────────────────────────────────
def _iter_gentl_interfaces(gentl):
    """
    yield (iface_index:int, iface_obj)  ← with-context 로 이미 감싸진 상태
    대응 가능한 API 패턴
    ─────────────────────────────────────────────────────────────
      ① get_num_interfaces / get_interface_id   (eGrabber 1.x~2.x)
      ② interfaces() 메서드                      (3.0 ~ 3.1.x)
      ③ interfaces  프로퍼티(list-like)          (3.2+)
      ④ enumerate_interfaces() 메서드            (드문 파생 build)
      ⑤ interface_ids()  메서드                 (또 다른 파생 build)
    """
    # ① legacy API: get_num_interfaces / get_interface_id
    if hasattr(gentl, "get_num_interfaces"):
        for i in range(gentl.get_num_interfaces()):
            iface_id = gentl.get_interface_id(i)
            with gentl.open_interface(iface_id) as iface:
                yield i, iface
        return

    # ② 3.0 – 3.1 : interfaces() 메서드
    if callable(getattr(gentl, "interfaces", None)):
        for i, iface in enumerate(gentl.interfaces()):       # list[Interface]
            with iface:
                yield i, iface
        return

    # ③ 3.2+ : interfaces 프로퍼티 (list-like)
    if hasattr(gentl, "interfaces"):
        for i, iface in enumerate(gentl.interfaces):         # property
            with iface:
                yield i, iface
        return

    # ④ enumerate_interfaces()  → interface ID 시퀀스
    if callable(getattr(gentl, "enumerate_interfaces", None)):
        for i, iface_id in enumerate(gentl.enumerate_interfaces()):
            with gentl.open_interface(iface_id) as iface:
                yield i, iface
        return

    # ⑤ interface_ids()  → interface ID 시퀀스
    if callable(getattr(gentl, "interface_ids", None)):
        for i, iface_id in enumerate(gentl.interface_ids()):
            with gentl.open_interface(iface_id) as iface:
                yield i, iface
        return

    # ⑥ 마지막 수단: EGenTL 객체 자체가 iterable 한 경우
    try:
        for i, iface in enumerate(gentl):
            with iface:
                yield i, iface
        return
    except TypeError:
        pass

    # 전부 실패 시 명시적 예외
    raise AttributeError("EGenTL API: 인터페이스 열거 메서드를 찾을 수 없습니다.")
# ------------------------------------------------------------------
#  안전하게 벤더/모델/시리얼을 읽어오는 헬퍼
# ------------------------------------------------------------------
def _safe_get_dev_info(grabber, key: str) -> str:
    """
    - eGrabber 1.x/2.x : grabber.get_info(key)
    - Coaxlink 파이썬 API : grabber.remote.get(key)
    - 모두 실패        : 'Unknown'
    """
    # ① 표준 get_info()
    if hasattr(grabber, "get_info"):
        try:
            val = grabber.get_info(key)
            if val:
                return str(val)
        except Exception:
            pass

    # ② GenICam remote 노드
    try:
        if hasattr(grabber, "remote") and grabber.remote:
            return str(grabber.remote.get(key))
    except Exception:
        pass

    return "Unknown"

def _probe_attr(obj, *candidates, default="Unknown"):
    """obj 에서 candidates 중 처음으로 존재하는 속성/메서드를 호출/읽어 반환."""
    for name in candidates:
        if hasattr(obj, name):
            attr = getattr(obj, name)
            try:
                return attr() if callable(attr) else attr
            except Exception:
                pass
    return default

# ---------------------------------------------------------------------
#  helper: grabber 가 가진 CameraPort 개수 추정
# ---------------------------------------------------------------------
def _num_camera_ports(grabber) -> int:
    """
    Coaxlink 계열 보드에서 사용 가능한 Camera Port 개수를 반환한다.
    드라이버/펌웨어 버전에 따라 노드 이름이 조금씩 달라질 수 있으므로
    여러 후보를 순차적으로 시도한다.
    """
    for key in ("NumCameraPorts", "CameraPortCount", "CameraPorts"):
        try:
            return int(grabber.remote.get(key))
        except Exception:
            pass
    return 1      # 알 수 없으면 1개라고 가정
# ----------------------------------------------------------------------
#  Grabber 1개 안에 몇 개의 Visible-Device(Port)가 존재하는지 조사
# ----------------------------------------------------------------------
def _iter_visible_devices(grabber):
    """
    yield (port_idx:int, model:str, serial:str, vendor:str)
    - VisibleDeviceSelector   : Coaxlink 3.x+
    - PortSelector / PortId   : 구 버전/다른 Producer
    """
    remote = getattr(grabber, "remote", None)
    if remote is None:
        return

    # ① Coaxlink 3.x : VisibleDeviceSelector (Enumeration)
    if "VisibleDeviceSelector" in remote.features():
        enum = remote.getEnum("VisibleDeviceSelector")   # ← egrabber.query.enum_entries(...)
        for idx, entry in enumerate(enum):
            remote.set("VisibleDeviceSelector", entry)
            yield idx, remote.get("DeviceModelName", str), \
                       remote.get("DeviceSerialNumber", str), \
                       remote.get("DeviceVendorName", str)
    # ② Fallback : PortSelector / PortId
    elif "PortSelector" in remote.features():
        max_ports = int(remote.get("PortCount"))
        for idx in range(max_ports):
            remote.set("PortSelector", idx)
            yield idx, remote.get("DeviceModelName", str), \
                       remote.get("DeviceSerialNumber", str), \
                       remote.get("DeviceVendorName", str)


#  CameraController 내부 ─────────────────────────────────────────────
from egrabber.discovery import EGrabberDiscovery, EGrabberCameraInfo, EGrabberInfo
from egrabber.generated.constants import DEVICE_ACCESS_READONLY as RO
import egrabber.generated.constants as C
OFFLINE_CODES = {
    C.GC_ERR_NOT_AVAILABLE,
    C.GC_ERR_INVALID_ADDRESS,
    C.GC_ERR_CUSTOM_DRIVER_NOT_AVAILABLE,
    C.GC_ERR_ERROR,  # == -1
}


def _is_online(cam_info: EGrabberCameraInfo) -> bool:
    """RO 플래그로 ‘살짝’ 열어 본다."""
    for flags in (C.DEVICE_ACCESS_CONTROL, RO):
        for remote in (True, False):
            try:
                with EGrabber(cam_info, device_open_flags=flags,
                              remote_required=remote):
                    return True
            except GenTLException as ge:
                if getattr(ge, "gc_err", None) in OFFLINE_CODES:
                    continue
            except Exception:
                # 파이썬 레벨 에러(속성 누락 등)는 무시
                pass
    return False


def _enum_entries_safe(dev, node: str) -> List[str]:
    try:
        if hasattr(dev, "get_enum_entries"):
            return list(dev.get_enum_entries(node))
        if hasattr(dev, "getEnum"):
            return list(dev.getEnum(node))
        if hasattr(dev, "getNode"):
            n = dev.getNode(node)
            if n:
                return [e.getSymbolic() for e in n.getEnumEntries() if e.isAvailable()]
    except Exception:
        pass
    return []

from workers.grab_worker import GrabWorker
_grab_worker: Optional[GrabWorker] = None
DEFAULT_STATS_INTERVAL = 0.5

def read_lost_triggers_legacy(dev) -> int:
    """
    FW < 4.38 보드에서 Trigger 누락을 추산하기 위한 에러 카운터 합산.
    ErrorSelector : {DidNotReceiveTriggerAck, CameraTriggerOverrun}
    """
    lost = 0
    if {"ErrorSelector", "ErrorCount"}.issubset(dev.features()):
        for sel in ("DidNotReceiveTriggerAck", "CameraTriggerOverrun"):
            try:
                dev.set("ErrorSelector", sel)
                lost += int(dev.get("ErrorCount"))
            except Exception:
                continue
    return lost

# ────────── Stream-Port Error 카운터 기반 Frame-Loss ──────────
NEW_ERROR_ENUMS = (
    "StreamPacketSizeError", "StreamPacketFifoOverflow",
    "CameraTriggerOverrun", "DidNotReceiveTriggerAck",
    "TriggerPacketRetryError", "InputStreamFifoHalfFull",
    "InputStreamFifoFull", "ImageHeaderError",
    "MigAxiWriteError", "MigAxiReadError",
    "PacketWithUnexpectedTag", "StreamPacketArbiterError",
    "StartOfScanSkipped", "PrematureEndOfScan",
    "ExternalTriggerReqsTooClose",
    "StreamPacketCrcError0", "StreamPacketCrcError1",
    "StreamPacketCrcError2", "StreamPacketCrcError3",
    "StreamPacketCrcError4", "StreamPacketCrcError5",
    "StreamPacketCrcError6", "StreamPacketCrcError7",
)

# ────────── Stream-Port Error 카운터 기반 Frame-Loss (개선판) ──────────
def read_frame_loss_stream(dev) -> int:
    """
    StreamPort-Errors 섹션에서 ErrorSelector/E​rrorCount를
    **모든 enum 엔트리**에 대해 누적해 반환한다.
    ErrorSelector 또는 ErrorCount 노드가 없으면 0.
    """
    lost = 0
    if {"ErrorSelector", "ErrorCount"}.issubset(dev.features()):
        # ▶︎ ErrorSelector 의 전체 enum 항목을 런타임에 확보
        for sel in _enum_entries_safe(dev, "ErrorSelector"):
            try:
                dev.set("ErrorSelector", sel)
                lost += int(dev.get("ErrorCount"))
            except Exception:
                continue     # 읽기 불가·권한 없음 등은 무시
    return lost



class CameraController(QObject):
    """
    개별 카메라 하드웨어를 제어하고 상태를 관리하는 핵심 클래스.
    """

    camera_connected    = pyqtSignal()
    camera_disconnected = pyqtSignal()
    frame_ready         = pyqtSignal(np.ndarray)
    reshaped_frame_ready = pyqtSignal(np.ndarray)

    # ──────────────────── ❶ __init__ 전체 (NEW 필드 포함) ────────────────────
    def __init__(self, enable_param_cache: bool = False):
        QObject.__init__(self)
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")

        # low-level handles & flags
        self.grabber: Optional[EGrabber] = None
        self._device: Optional[Any] = None  # ★ NEW – 내부 Device-module
        self.params: Optional[Any] = None
        self.connected: bool = False
        self.grabbing: bool = False
        self._lock = threading.RLock()
        self._last_buffer: Optional[Any] = None
        self._last_np_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        # identifiers
        self.cam_id: str = f"cam_{id(self)}"
        self.serial_number: str = "N/A"
        self.grabber_index: int = -1

        # SDK helpers
        self.gentl = _get_gentl()

        # NEW – last grabbed buffer cache
        self._last_buffer: Optional[Any] = None

        # parameter-cache option
        self.enable_param_cache = enable_param_cache
        self._param_cache: Dict[str, Any] = {}
        self._param_cache_timestamp: Dict[str, float] = {}
        self.cache_timeout_sec = 1.0

        # runtime stats
        self._stats_interval: float = DEFAULT_STATS_INTERVAL
        now = time.time()
        self.stats = {
            "frame_count": 0,
            "error_count": 0,
            "last_frame_time": now,
            "start_time": 0.0,
            "fps": 0.0,
            "frame_loss": 0.0,
        }
        self._last_stats_update = now

        # misc flags / workers
        self._memento_live: bool = False
        self._grab_thr: Optional["_GrabThread"] = None
        self._live_view: bool = False
        # __init__ 내부 어딘가에 추가
        self._buffers_allocated: bool = False
        self._announced_buffers: int = 0

    # ──────────────────────── END __init__ ────────────────────────────────

    def set_last_buffer(self, buffer: Any) -> None:
        """
        [스레드 안전] 실행 중 가장 마지막에 획득한 **eg.Buffer** 객체를 저장합니다.
        주로 예외 발생 시 오류 이미지 저장을 위해 사용됩니다.
        """
        with self._frame_lock:
            self._last_buffer = buffer


    def get_last_buffer(self) -> Optional[Any]:
        """
        [스레드 안전] `set_last_buffer()`로 저장된 **eg.Buffer** 를 반환합니다.
        """
        with self._frame_lock:
            return self._last_buffer

    def set_last_np_frame(self, frame: np.ndarray) -> None:
        """
        [스레드 안전] 가장 최근에 성공적으로 디코딩된 NumPy 프레임을 캐시에 저장합니다.
        GrabWorker가 이 메서드를 호출합니다.
        """
        with self._frame_lock:
            self._last_np_frame = frame

    def get_last_np_frame(self) -> Optional[np.ndarray]:
        """
        [스레드 안전] 캐시된 최신 NumPy 프레임의 복사본을 반환합니다.
        액션(get_last_cached_frame 등)이 이 메서드를 호출합니다.
        """
        with self._frame_lock:
            return self._last_np_frame.copy() if self._last_np_frame is not None else None
    # ───────────────────────────── live-view helpers ────────────
    def start_live_view(self, *, buffer_count: int = 32, max_fps: int = 30) -> None:
        """
        Enable real-time preview via GrabWorker.
        - Sets _live_view = True
        - Starts grab (if needed) with a healthy buffer count (>= 32)
        - Registers controller in the global GrabWorker singleton
        """
        if not self.connected:
            self.logger.warning("[%s] Cannot start live view: not connected.", self.cam_id)
            return
        if self._live_view:
            self.logger.debug("[%s] Live view is already active.", self.cam_id)
            return

        self._live_view = True
        # If not already grabbing, start with a generous buffer count for live view
        if not self.is_grabbing():
            self.start_grab(buffer_count=buffer_count, min_buffers=32)

        self._ensure_grab_worker(max_fps=max_fps)
        self.logger.info("[%s] Live-view session started.", self.cam_id)

    def stop_live_view(self) -> None:
        """Disable preview and stop acquisition cleanly."""
        if not self._live_view:
            return
        self._live_view = False
        self._detach_from_grab_worker()
        if self.is_grabbing():
            self.stop_grab(flush=False)  # GRACEFUL 정지
        self.logger.info("[%s] Live-view session stopped.", self.cam_id)

    @property
    def device(self):
        """Grabber Device-module 핸들(read-only). None일 수 있음."""
        return self._device

    # ─────────────────────────────────────────────────────────────
    #  Legacy grab control aliases
    # ─────────────────────────────────────────────────────────────
    def grab_start(
            self,
            buffer_count: Optional[int] = None,
            *,
            min_buffers: Optional[int] = None,
            align_bytes: int = 4096,
            force_realloc: bool = False,
    ) -> None:
        """
        **레거시 호환용** : 내부적으로 ``start_grab()`` 을 그대로 호출합니다.

        예전 코드에서
            ctrl.grab_start(...)
        로 부르던 부분을 그대로 유지할 수 있습니다.
        """
        return self.start_grab(
            buffer_count,
            min_buffers=min_buffers,
            align_bytes=align_bytes,
            force_realloc=force_realloc,
        )
    def flush_buffers(self, *, block: bool = False, timeout: float = 0.5) -> None:
        """
        DMA FIFO / pending-buffer 큐를 비웁니다.

        block=True 이면 큐가 0개가 될 때까지(최대 *timeout*) active-wait.
        """
        if not self.grabber:             # 연결 안 됐으면 no-op
            return

        log = self.logger
        try:
            # SDK 23.12+ : flush_buffers()
            if hasattr(self.grabber, "flush_buffers"):
                self.grabber.flush_buffers()
            # 구버전 low-level API
            elif hasattr(self.grabber, "dma_write_flush"):
                self.grabber.dma_write_flush()
        except Exception as exc:         # 실제 오류는 디버그용으로만 남김
            log.debug("flush_buffers ignored error: %s", exc)

        if block:
            deadline = time.time() + timeout
            while self._pending_buffer_count() > 0 and time.time() < deadline:
                time.sleep(0.001)

    def wait_until_idle(self, timeout: float = 0.5) -> None:
        """
        AcquisitionStop 이후 grabber 상태가 idle 로 바뀔 때까지 대기합니다.
        """
        deadline = time.time() + timeout
        while self.is_grabbing() and time.time() < deadline:
            time.sleep(0.001)

    def _stop_grab_safe(self, grace_ms: int = 50) -> None:
        """
        (내부용) AcquisitionStop 후 DataStream을 정지시키는 안전한 중지.
        - grab 중이 아닐 때는 즉시 반환.
        - `flush=False`로 `stop_grab`을 호출하여 버퍼를 소모하며 종료.
        """
        if not self.is_grabbing():
            return
        # Use the main robust stop_grab method with flush=False for graceful shutdown
        self.stop_grab(flush=False)
        time.sleep(grace_ms / 1000.0)


    def grab_stop(self) -> None:
        """
        **레거시 호환용** : 내부적으로 ``stop_grab()`` 을 그대로 호출합니다.

        예전 코드에서
            ctrl.grab_stop()
        로 부르던 부분을 그대로 유지할 수 있습니다.
        """
        self.stop_grab()

    # ───────────────── Internal GrabWorker (singleton) Management ─────────────────
    def _ensure_grab_worker(self, *, max_fps: int = 30) -> None:
        """Create or update the global GrabWorker when live-view is ON."""
        global _grab_worker
        if _grab_worker is None or not _grab_worker.isRunning():
            _grab_worker = GrabWorker(controllers=[self], max_fps=max_fps)
            _grab_worker.start()
            self.logger.debug("GrabWorker started (singleton)")
        elif self not in _grab_worker.controllers:
            _grab_worker.add_controller(self)
            self.logger.debug("GrabWorker » added %s", self.cam_id)


    def _detach_from_grab_worker(self) -> None:
        """Remove this controller from GrabWorker; stop thread if empty."""
        global _grab_worker
        if _grab_worker and self in _grab_worker.controllers:
            _grab_worker.remove_controller(self)
            if not _grab_worker.controllers:
                _grab_worker.stop()
                _grab_worker = None
                self.logger.debug("GrabWorker terminated (no controllers)")



    def setup_camera_for_hw_trigger(
            self,
            trigger_selector: Literal["ExposureStart", "FrameStart"] = "ExposureStart",
            trigger_source: Literal["LinkTrigger0", "Line1"] = "LinkTrigger0",
            activation: Literal["RisingEdge", "FallingEdge"] = "RisingEdge",
    ) -> None:
        self._check_connected()
        dev = self.params
        log = self.logger
        log.info(
            "[%s] 카메라 HW-Trigger 설정 시작  Selector=%s  Source=%s",
            self.cam_id, trigger_selector, trigger_source,
        )
        with suppress(Exception):
            self.execute_command("AcquisitionStop")
        time.sleep(0.05)
        if "TriggerSelector" in dev.features():
            dev.set("TriggerSelector", trigger_selector)
        if "TriggerMode" in dev.features():
            dev.set("TriggerMode", "Off")
        if "TriggerSource" in dev.features():
            dev.set("TriggerSource", trigger_source)
        if "TriggerActivation" in dev.features():
            dev.set("TriggerActivation", activation)
        if "TriggerMode" in dev.features():
            dev.set("TriggerMode", "On")
        log.info("[%s] 카메라 HW-Trigger 설정 완료", self.cam_id)

    # ────────────────────────────── Memento control ──────────────────────
    def toggle_memento(self, enable: bool, ring_mb: int = 256) -> None:
        with self._lock:  # Thread-safe
            if not EURESYS_AVAILABLE:
                self.logger.info(f"[{self.cam_id}] toggle_memento ignored – eGrabber unavailable.")
                return
            if enable == self._memento_live:
                return
            if self.grabber_index < 0:
                self.logger.warning(f"[{self.cam_id}] Grabber index unknown – toggle skipped.")
                return
            try:
                cmd = [
                    "ringbuffer",
                    "--enable" if enable else "--disable",
                    f"--grabber={self.grabber_index}",
                ]
                if enable:
                    cmd.append(f"--ringbuffer={ring_mb}m")
                _run_memento_cmd(cmd)
                self._memento_live = enable
                self.logger.info(f"[{self.cam_id}] Memento Live {'ON' if enable else 'OFF'} (grabber={self.grabber_index})")
            except FileNotFoundError:
                self.logger.warning(f"[{self.cam_id}] Memento CLI not found – toggle skipped.")
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"[{self.cam_id}] Memento toggle failed: {e}")

    def connect_camera_by_info(self, cam_info: EGrabberCameraInfo) -> bool:
        from egrabber.generated.constants import DEVICE_ACCESS_CONTROL
        with self._lock:
            if self.connected:
                return True
            try:
                self.grabber = EGrabber(cam_info,
                                        device_open_flags=DEVICE_ACCESS_CONTROL)

                self.params = self.grabber.remote
                self._device = getattr(self.grabber, "device", None)  # ★ 수정

                gi = cam_info.grabbers[0]
                self.serial_number = gi.deviceSerialNumber or \
                                     f"SN_{gi.interface_index}_{gi.device_index}"
                self.cam_id = f"{gi.deviceModelName}-{self.serial_number}"
                self.grabber_index = int(
                    getattr(self.grabber, "get_info", lambda k: -1)("GrabberIndex") or -1
                )
                self.connected = True
                self.logger.info("[%s] CONNECT OK (DEVICE_ACCESS_CONTROL)", self.cam_id)
                return True
            except Exception as e:
                self.logger.error("connect_camera_by_info failed: %s", e, exc_info=True)
                self.disconnect_camera()
                return False


    # ------------------------------------------------------------------
    #  노드 존재 여부를 결정 – _find_nodemap() 재사용
    # ------------------------------------------------------------------
    def _feature_exists(self, name: str) -> bool:
        """
        어떤 레벨(nodemap)에 있든 *name* 을 노출하면 True.
        """
        return self._find_nodemap(name) is not None



    def setup_grabber_for_hw_trigger(
            self,
            *,  # keyword-only
            line_tool: str = "LIN1",
            link_trigger: str = "LinkTrigger0",
            edge: str = "RisingEdge",
            camera_control_method: str = "RC",
            cycle_trigger_source: str = "Immediate",
            cycle_period_us: float = 3360.0,
            idempotent: bool = True,
            verify: bool = True,
    ) -> None:
        from core.camera_exceptions import GrabberError

        dev = getattr(self.grabber, "device", None)
        if dev is None:  # 멀티-카메라 세컨드
            self.logger.warning("[%s] grabber.device unavailable – CIC setup skipped (shared grabber)",
                                self.cam_id)
            return

        feat = dev.features()
        f_exists = feat.__contains__

        def safe_set(n, v):
            try:
                if not f_exists(n):
                    return False
                if idempotent and str(dev.get(n)) == str(v):
                    return True
                dev.set(n, v)
                return True
            except Exception:  # 읽기 전용 등
                return False

        # ── ① FW 4.38+ : CycleTriggerSource 가 존재 ──────────────────
        if f_exists("CycleTriggerSource"):
            safe_set("CameraControlMethod", camera_control_method)
            if f_exists("CxpTriggerMessageSelector"):
                safe_set("CxpTriggerMessageSelector", link_trigger)
                safe_set("CxpTriggerMessageSource", "CycleTrigger")

            use_cg = cycle_trigger_source.lower() in ("immediate", "startcycle", "cyclegenerator0")
            if f_exists("DeviceLinkTriggerToolSelector"):
                safe_set("DeviceLinkTriggerToolSelector", link_trigger)
                safe_set("DeviceLinkTriggerToolSource",
                         "CycleGenerator0" if use_cg else line_tool)
                safe_set("DeviceLinkTriggerToolActivation", edge)

            safe_set("CycleTriggerSource", cycle_trigger_source)
            if use_cg and f_exists("CycleTriggerPeriod"):
                safe_set("CycleTriggerPeriod", float(cycle_period_us))

            # optional sanity
            if verify and f_exists("CycleLostTriggerCount"):
                if int(dev.get("CycleLostTriggerCount")):
                    raise GrabberError(f"[{self.cam_id}] non-zero CycleLostTriggerCount")

            self.logger.info("[%s] CIC ready (FW≥4.38, CTS=%s, Period=%.3f µs)",
                             self.cam_id, cycle_trigger_source, cycle_period_us)
            return

        # ── ② 구형 FW : DLT Toolbox 만 존재 ────────────────────────────
        dlt_nodes = {"DeviceLinkTriggerToolSelector",
                     "DeviceLinkTriggerToolSource",
                     "DeviceLinkTriggerToolActivation"}
        if dlt_nodes.issubset(feat):
            safe_set("DeviceLinkTriggerToolSelector", link_trigger)
            safe_set("DeviceLinkTriggerToolSource", line_tool)
            safe_set("DeviceLinkTriggerToolActivation", edge)
            self.logger.info("[%s] DLT mapping %s→%s (%s) – CIC not available",
                             self.cam_id, line_tool, link_trigger, edge)
            return

        # ── ③ 아무 경로도 없으면 예외 ─────────────────────────────────
        raise GrabberError(f"[{self.cam_id}] Neither CIC nor DLT nodes present – check firmware")


    # ───────────────────── NEW: CIC Counter 헬퍼 & 롱런 시험 ───────────────
    def reset_cic_counters(self) -> None:
        """CycleLostTriggerCount 등을 0으로 리셋."""
        if self._feature_exists("CycleLostTriggerCountReset"):
            self.grabber.device.run("CycleLostTriggerCountReset")

    def read_cic_lost(self) -> int:
        """현재 누락된 트리거 수(노드 없으면 0)."""
        if self._feature_exists("CycleLostTriggerCount"):
            return int(self.grabber.device.get("CycleLostTriggerCount"))
        return 0

    def trigger_integrity_test(
            self,
            expected_triggers: int,
            poll_interval: float = 5.0,
    ) -> None:
        """
        멀티-카메라 환경에서 HW-Trigger 롱런 무-누락 시험.
        누락 발생 시 GrabberError 예외로 즉시 중단.
        """
        from core.camera_exceptions import GrabberError

        self.reset_cic_counters()
        self.start_grab()
        recv = 0
        t_last = time.time()

        while recv < expected_triggers:
            recv = self.get_received_frame_count()
            if time.time() - t_last >= poll_interval:
                lost = self.read_cic_lost()
                self.logger.info("[%s] progress %d/%d, lost=%d",
                                 self.cam_id, recv, expected_triggers, lost)
                if lost:
                    raise GrabberError(
                        f"[{self.cam_id}] 롱런 시험 실패 – 누락 {lost} (수신 {recv})"
                    )
                t_last = time.time()
            time.sleep(0.002)

        self.logger.info("[%s] 롱런 시험 통과 – 누락 0 (수신 %d)", self.cam_id, recv)

    def get_received_frame_count(self) -> int:
        """Frames successfully grabbed since the last start."""
        return int(self.stats.get("frame_count", 0))

    # ───────────────────── PATCH: enum helper (3.7+ safe) ────────────────
    def _enum_entries(self, node_name: str) -> List[str]:
        """주어진 Enumeration 노드의 모든 엔트리 이름을 반환."""
        return self.grabber.device.get_enum_entries(node_name)


    def stop_grab_flush(self) -> None:
        """즉시 중단 + flush_buffers() ― 사용 시 크래시 위험 주의!"""
        self.stop_grab(flush=True)

    def discover_cameras(self) -> List[Dict[str, Any]]:
        """
        카메라-centric 스캔 (EGrabberDiscovery.cameras 활용).
        """
        with self._lock:
            cams: List[Dict[str, Any]] = []
            try:
                disc = EGrabberDiscovery(self.gentl)
                disc.discover(find_cameras=True)

                # cameras 는 len/[] 만 지원 → range() 로 인덱스 반복
                for idx in range(len(disc.cameras)):
                    cam_info: EGrabberCameraInfo = disc.cameras[idx]

                    # 첫 grabber(=포트0) 로부터 인덱스·모델·시리얼 추출
                    gi = cam_info.grabbers[0]           # type: EGrabberInfo

                    online = _is_online(cam_info)       # 빠른 alive check
                    self.logger.debug("[%s] cam#%d (%d/%d/%d) online=%s",
                                      self.cam_id, idx,
                                      gi.interface_index, gi.device_index,
                                      gi.stream_index, online)

                    cams.append({
                        "camera_info": cam_info,
                        "iface_idx":   gi.interface_index,
                        "dev_idx":     gi.device_index,
                        "stream_idx":  gi.stream_index,
                        "vendor":      gi.deviceVendorName,
                        "model":       gi.deviceModelName,
                        "serial":      gi.deviceSerialNumber,
                        "online":      online,
                    })
            except Exception as e:
                self.logger.error("[%s] discovery failed: %s",
                                  self.cam_id, e, exc_info=True)

            self.logger.info("[%s] Discovered %d camera(s) (raw)",
                             self.cam_id, len(cams))
            return cams

    @staticmethod
    def connect_all(enable_param_cache: bool = False) -> List['CameraController']:
        controller_pool.flush(disconnect=True)
        try:
            discovery = EGrabberDiscovery(_get_gentl())
            discovery.discover(find_cameras=True)
            camera_infos = [discovery.cameras[i] for i in range(len(discovery.cameras))]
        except Exception as e:
            logger.error(f"Camera discovery failed: {e}", exc_info=True)
            return []

        connected_controllers: List[CameraController] = []
        for cam_info in camera_infos:
            if _is_online(cam_info):
                ctrl = CameraController(enable_param_cache=enable_param_cache)
                if ctrl.connect_camera_by_info(cam_info):
                    controller_pool.register(ctrl)
                    connected_controllers.append(ctrl)
        return connected_controllers
    def setup_for_hardware_trigger(self, source: str = "LinkTrigger0") -> None:
        """
        Legacy API wrapper. 외부 코드 호환성을 위해 유지.
        """
        self.setup_camera_for_hw_trigger(
            trigger_selector="ExposureStart",
            trigger_source=source,
            activation="RisingEdge",
        )
    # ------------------------------------------------------------------
    #  Grabber CIC (내부 Trigger Generator) 활성화 – 안정판
    # ------------------------------------------------------------------
    def _enable_internal_grabber_trigger(
            self,
            *,

            link_trigger: str = "LinkTrigger0",
            min_period_us: float = 3.36,
            cycle_source: str = "Immediate",
            trigger_activation: str = "RisingEdge",
            verify: bool = True,
    ) -> bool:
        """
        Grabber(Device-module) CIC 을 활성화한다.

        Parameters
        ----------
        link_trigger : str
            LinkTriggerN 에 매핑할 대상.
        min_period_us : float
            원하는 최소 주기(µs). 하드웨어 최소값보다 작으면 클램프.
        cycle_source : str
            CycleTriggerSource 노드 값.
        trigger_activation : str
            DeviceLinkTriggerToolActivation 값(RisingEdge/FallingEdge).
        verify : bool
            CycleLostTriggerCount 검사 여부.
        """
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            self.logger.warning("[%s] grabber.device unavailable – CIC enable skipped",
                                self.cam_id)
            return False

        feats = dev.features()
        exists = feats.__contains__

        required = {"CameraControlMethod", "CycleTriggerSource", "CycleTriggerPeriod",
                    "CxpTriggerMessageSelector", "CxpTriggerMessageSource",
                    "DeviceLinkTriggerToolSelector", "DeviceLinkTriggerToolSource",
                    "DeviceLinkTriggerToolActivation"}
        if not required.issubset(feats):
            self.logger.warning("[%s] CIC nodes missing: %s – falling back",
                                self.cam_id, ", ".join(sorted(required - feats)))
            return False

        def safe_set(node: str, value) -> None:
            if not exists(node):
                return
            try:
                if str(dev.get(node)) != str(value):
                    dev.set(node, value)
            except Exception as exc:          # RW 안 되는 경우 등
                self.logger.debug("[%s] %s write failed: %s", self.cam_id, node, exc)

        # 실제 설정 -------------------------------------------------------
        safe_set("CameraControlMethod", "RC")
        safe_set("CxpTriggerMessageSelector", link_trigger)
        safe_set("CxpTriggerMessageSource", "CycleTrigger")
        safe_set("DeviceLinkTriggerToolSelector", link_trigger)
        safe_set("DeviceLinkTriggerToolSource", "CycleGenerator0")
        safe_set("DeviceLinkTriggerToolActivation", trigger_activation)

        if exists("CycleMinimumPeriod"):
            hw_min = float(dev.get("CycleMinimumPeriod"))
            if min_period_us < hw_min:
                self.logger.debug("[%s] min_period %.3f → %.3f (hw min)",
                                  self.cam_id, min_period_us, hw_min)
                min_period_us = hw_min
        safe_set("CycleTriggerSource", cycle_source)
        safe_set("CycleTriggerPeriod", float(min_period_us))

        # 검증 ------------------------------------------------------------
        if verify and exists("CycleLostTriggerCount"):
            lost = int(dev.get("CycleLostTriggerCount"))
            if lost:
                raise ParameterError(f"[{self.cam_id}] CIC lost-trigger ≠ 0 ({lost})")

        self.logger.info("[%s] CIC enabled → LT=%s  Period=%.3f µs  Src=%s  Act=%s",
                         self.cam_id, link_trigger, min_period_us,
                         cycle_source, trigger_activation)
        return True


    def enable_internal_grabber_trigger(self, **kwargs) -> bool:  # noqa: D401
        """Thin wrapper around :py:meth:`_enable_internal_grabber_trigger`."""
        return self._enable_internal_grabber_trigger(**kwargs)

    def execute_grabber_cycle_trigger(self) -> None:
        """
        Grabber(Device-module) 쪽에서 **하나의 트리거 펄스**를 발생시킨다.
        • FW ≥ 4.38  :  StartCycle  노드
        • 구형 FW     :  CxpTriggerMessageSend  +  LinkTrigger0
        """
        from core.camera_exceptions import CommandExecutionError

        dev = getattr(self.grabber, "device", None)
        if not dev:
            raise CommandExecutionError("grabber.device is None")

        # 신형 FW – StartCycle
        if "StartCycle" in dev.features():
            dev.execute("StartCycle")
            self.logger.debug("[%s] Device.StartCycle executed", self.cam_id)
            return

        # 구형 FW – CXP Trigger Message
        if "CxpTriggerMessageSend" in dev.features():
            with suppress(Exception):
                if "CxpTriggerMessageID" in dev.features():
                    dev.set("CxpTriggerMessageID", 0)
            dev.execute("CxpTriggerMessageSend")
            self.logger.debug("[%s] CxpTriggerMessageSend executed", self.cam_id)
            return

        raise CommandExecutionError("No cycle-trigger command available in Device-module")

    def connect_camera(self) -> bool:
        """Connect to the first available grabber (generic path)."""
        with self._lock:
            if self.connected:
                self.logger.warning("[%s] Already connected.", self.cam_id)
                return True
            try:
                self.logger.debug("[%s] Initializing EGrabber…", self.cam_id)
                self.grabber = EGrabber(self.gentl)
                if not getattr(self.grabber, "remote", None):
                    raise CameraConnectionError("remote interface unavailable")

                # GenICam nodemaps
                self.params = self.grabber.remote
                self._device = getattr(self.grabber, "device", None)  # ★ 수정
                if self._device is None:
                    with suppress(GenTLException, AttributeError):
                        self._device = self.grabber._get_device_module()  # type: ignore
                if self._device is None:
                    self.logger.warning("[%s] Device-module missing ⇒ CIC 제한", self.cam_id)

                # IDs
                self.serial_number = self.get_device_serial_number()
                self.cam_id = self.serial_number or self.cam_id
                gi = getattr(self.grabber, "get_info", lambda k: -1)
                self.grabber_index = int(gi("GrabberIndex") or -1)

                # state
                self.connected, self.grabbing = True, False
                controller_pool.register(self)
                self.camera_connected.emit()
                self.logger.info("[%s] CONNECT OK", self.cam_id)
                return True

            except Exception as e:
                self.logger.error("[%s] connect_camera failed: %s",
                                  self.cam_id, e, exc_info=True)
                self.disconnect_camera()
                raise CameraConnectionError(f"[{self.cam_id}] {e}") from e



    def configure_trigger(self, mode: str, source: str,
                          activation: str = "RisingEdge") -> None:
        self._check_connected()
        p = self.params
        self.logger.info("[%s] 트리거 설정 시작: Mode=%s, Source=%s", self.cam_id, mode, source)

        if "TriggerSelector" in p.features() and self.is_writeable("TriggerSelector"):
            avail = self.get_enumeration_entries("TriggerSelector")
            selector = next((s for s in ("FrameStart", "ExposureStart") if s in avail),
                            avail[0] if avail else None)
            if selector:
                self.set_param("TriggerSelector", selector)

        if self.is_writeable("TriggerMode"):
            self.set_param("TriggerMode", "Off")
        else:
            raise ParameterSetError("TriggerMode를 Off로 설정 불가")

        if mode == "On":
            if self.is_writeable("TriggerSource"):
                if source not in self.get_enumeration_entries("TriggerSource"):
                    raise ParameterSetError(
                        f"[{self.cam_id}] TriggerSource '{source}' not supported "
                        f"({self.get_enumeration_entries('TriggerSource')})"
                    )
                self.set_param("TriggerSource", source)
            if self.is_writeable("TriggerActivation"):
                self.set_param("TriggerActivation", activation)

        if self.is_writeable("TriggerMode"):
            self.set_param("TriggerMode", mode)

        self.logger.info("[%s] 트리거 설정 완료", self.cam_id)

    def execute_command(self, node_name: str, timeout: float = 1.0) -> None:
        """
        Executes a command node, searching through all available nodemaps.
        Search order: camera-remote -> device-module -> grabber-remote.
        This version is more robust against API errors during node discovery.
        """
        self._check_connected()

        # Define the search path for nodemaps
        node_maps = [
            self.params,  # Camera remote (most common)
            getattr(self.grabber, "device", None),  # Grabber device module (for CIC, etc.)
            getattr(self.grabber, "remote", None),  # Grabber remote module
        ]

        for nm in filter(None, node_maps):
            # --- Strategy 1: Use GenApi Node interface if available (more robust) ---
            if hasattr(nm, "getNode"):
                with suppress(Exception):  # getNode can fail on some nodemaps
                    node = nm.getNode(node_name)
                    # Check if the node is valid, a command, and executable
                    if node and node.isFeature() and node.isWritable():
                        if hasattr(node, "execute"):  # Verify it's a command
                            node.execute()
                            if timeout > 0 and hasattr(node, 'wait_until_done'):
                                node.wait_until_done(timeout)
                            self.logger.info(
                                "[%s] Command '%s' executed via Node-API on %s.",
                                self.cam_id, node_name, type(nm).__name__
                            )
                            return

            # --- Strategy 2: Fallback to raw execute() method (for simpler interfaces) ---
            if hasattr(nm, "execute"):
                # This check ensures the feature actually exists before trying to execute it
                if hasattr(nm, "features") and node_name in nm.features():
                    try:
                        nm.execute(node_name)
                        self.logger.info(
                            "[%s] Command '%s' executed via raw execute() on %s.",
                            self.cam_id, node_name, type(nm).__name__
                        )
                        return
                    except Exception:
                        # This attempt failed, continue to the next nodemap
                        pass

        # If all attempts fail
        raise CommandExecutionError(
            f"[{self.cam_id}] Command '{node_name}' not found or not writable in any available nodemap."
        )
    def disconnect_camera(self) -> None:
        """연결을 해제합니다. 풀에서의 등록 해제는 flush가 담당합니다."""
        with self._lock:
            if not self.connected:
                return
            try:
                # Use flush=True to ensure a quick and clean disconnection.
                if self.is_grabbing() or self._live_view:
                    self.stop_live_view()
            finally:
                self.grabber = None
                self.params = None
                self._device = None
                self.connected = False
                self.grabbing = False
                self.logger.info(f"[{self.cam_id}] Camera resources released.")
    # ────────────────────────────── Grab control ─────────────────────────
    def is_grabbing(self) -> bool:
        """
        Check if the camera is currently grabbing frames.
        """
        with self._lock:
            if not self.connected or not self.grabber:
                return False
            # EGrabber의 is_grabbing 메서드 호출, 없으면 self.grabbing 반환
            return getattr(self.grabber, "is_grabbing", lambda: self.grabbing)()
    def get_grabber(self):
        """
        Return the underlying EGrabber instance.

        Added for backward-compatibility with legacy action modules that
        expect ctrl.get_grabber() to exist.
        """
        with self._lock:
            return self.grabber

    def _pending_buffer_count(self) -> int:
        """Best-effort pending buffer count from grabber.remote."""
        gm = getattr(self.grabber, "remote", None)
        if gm and hasattr(gm, "get_num_pending_buffers"):
            try:
                return int(gm.get_num_pending_buffers())
            except Exception:
                pass
        return 0

    # ==================================================================
    #  Grab start-stop
    def start_grab(
            self,
            buffer_count: Optional[int] = None,
            *,
            min_buffers: Optional[int] = None,
            align_bytes: int = 4096,
            force_realloc: bool = False,
    ) -> None:
        with self._lock:
            self._check_connected()

            if self.is_grabbing():
                self.logger.debug("[%s] Start grab ignored: already grabbing.", self.cam_id)
                return

            # ScanType 확인(생략)

            # 필요한 버퍼 수
            req_buffers = max(int(min_buffers or buffer_count or 8), 4)

            # announced buffer 수 읽기 (실패해도 재할당 강제 X)
            announced = None
            try:
                announced = int(getattr(self.grabber, "announced_buffer_count", lambda: 0)())
            except Exception as e:
                self.logger.debug("[%s] announced_buffer_count unavailable: %s", self.cam_id, e)

            if announced is not None:
                self._announced_buffers = announced
                self._buffers_allocated = announced > 0

            # 재할당 필요 여부
            need_realloc = bool(force_realloc)
            if not need_realloc:
                if not self._buffers_allocated:
                    need_realloc = True
                elif self._announced_buffers < req_buffers:
                    need_realloc = True

            # 필요시에만 재할당
            if need_realloc:
                self.logger.info("[%s] Reallocating %d buffers (align=%d)...",
                                 self.cam_id, req_buffers, align_bytes)
                realloc_func = getattr(self.grabber, "realloc_buffers", None)
                if not realloc_func:
                    raise AttributeError("realloc_buffers API missing on grabber")
                try:
                    realloc_func(req_buffers, alignment=align_bytes)
                except TypeError:
                    realloc_func(req_buffers)
                self._buffers_allocated = True
                self._announced_buffers = req_buffers
                self.logger.info("[%s] Buffers reallocated.", self.cam_id)

            # 스트림 정리 후 시작
            with suppress(Exception):
                getattr(self.grabber, "flush_buffers", lambda: None)()
            self.grabber.start()
            self.grabbing = True
            # 통계 초기화(기존 코드 그대로)

    # ==================================================================
    #  Grab stop (UPDATED – DataStream → Camera → Grabber 순 정지)
    # ==================================================================
    def stop_grab(self, *, flush: bool = False) -> None:
        with self._lock:
            if not self.is_grabbing():
                self._live_view = False
                return

            self.logger.debug("[%s] Stopping acquisition (flush=%s)...", self.cam_id, flush)

            # 1) DataStream stop
            try:
                import egrabber as _eg
                ds_flags = getattr(_eg, "DSStopFlags", None)
                stream = getattr(self.grabber, "stream", None)
                if ds_flags and stream:
                    stream.stop(ds_flags.ASYNC_ABORT if flush else ds_flags.ASYNC_GRACEFUL)
            except Exception as exc:
                self.logger.warning("[%s] DataStream stop failed: %s", self.cam_id, exc)

            # 2) Camera AcquisitionStop
            with suppress(Exception):
                self.execute_command("AcquisitionStop", timeout=0.5)

            # 3) Grabber.stop()
            with suppress(Exception):
                self.grabber.stop()

            # 4) Flush / revoke
            if flush:
                with suppress(Exception):
                    getattr(self.grabber, "flush_buffers", lambda: None)()
                with suppress(Exception):
                    getattr(self.grabber, "revokeAllBuffers", lambda: None)()
                # revokeAllBuffers를 한 경우엔 다음 시작 때 재할당 필요
                self._buffers_allocated = False
                self._announced_buffers = 0

            # 5) GrabWorker 분리(기존 코드)
            # 6) 상태 업데이트
            self.grabbing = False
            self._live_view = False

            # **추가: idle 대기 (busy 회피)**
            time.sleep(0.02)

    # ──────────────────────────────────────────────────────────────
    #  DMA 큐에 남아 있는 버퍼 1개를 버리고 파이프라인을 해제한다
    # ──────────────────────────────────────────────────────────────
    def _flush_one_buffer(self) -> None:
        """
        Output-큐에 남은 버퍼 하나를 pop 하여 드레인합니다.
        큐가 비어 있으면 GC_ERR_TIMEOUT 예외가 뜨는데, 이는 무시합니다.
        """
        try:
            # timeout=0 → 즉시 반환, 없으면 TimeoutException
            self.grabber.pop(0)          # type: ignore[attr-defined]
        except (TimeoutException, AttributeError):
            pass   # 이미 비어 있으면 OK

        # --- In camera_controller.py ---

        # In camera_controller.py

    def get_next_frame(
            self,
            timeout_ms: int = 3000,
            *,
            count_timeout_error: bool = True,
    ) -> Optional[np.ndarray]:
        """
        [수정됨] Robustly fetches the next frame, decodes it into a NumPy array,
        and correctly handles various pixel formats including 12/16-bit Bayer formats.
        """
        if not self.is_grabbing():
            raise GrabberNotActiveError(f"[{self.cam_id}] Cannot get frame: not grabbing.")

        try:
            with self._lock:
                with Buffer(self.grabber, timeout=timeout_ms) as buf:
                    # 1. 원시 eGrabber 버퍼 캐시
                    self.set_last_buffer(buf)

                    # 2. 버퍼 메타데이터 추출
                    from ctypes import POINTER, c_ubyte, cast
                    ptr = buf.get_info(BUFFER_INFO_BASE, INFO_DATATYPE_PTR)
                    w = buf.get_info(BUFFER_INFO_WIDTH, INFO_DATATYPE_SIZET)
                    h = buf.get_info(BUFFER_INFO_DELIVERED_IMAGEHEIGHT, INFO_DATATYPE_SIZET)
                    sz = buf.get_info(BUFFER_INFO_DATA_SIZE, INFO_DATATYPE_SIZET)

                    if not all((ptr, w, h, sz)):
                        raise FrameAcquisitionError(f"Invalid buffer metadata: ptr={ptr}, w={w}, h={h}, sz={sz}")

                    # 3. 픽셀 포맷 및 데이터 타입 결정 (캐시 없이 강제 리로드)
                    pix_fmt = self.get_param("PixelFormat", force_reload=True) or "Mono8"

                    # [핵심 수정] 12/16비트 Bayer 포맷을 처리하도록 pf_map 확장
                    # (데이터타입, 채널 수)
                    pf_map: dict[str, tuple[np.dtype, int]] = {
                        "Mono8": (np.uint8, 1),
                        "Mono10": (np.uint16, 1),
                        "Mono12": (np.uint16, 1),
                        "Mono16": (np.uint16, 1),
                        "RGB8": (np.uint8, 3),
                        "BGR8": (np.uint8, 3),
                        "RGB10": (np.uint16, 3),
                        "RGB12": (np.uint16, 3),
                        "RGB16": (np.uint16, 3),
                        # 8-bit Bayer
                        "BayerRG8": (np.uint8, 1),
                        "BayerGB8": (np.uint8, 1),
                        "BayerGR8": (np.uint8, 1),
                        "BayerBG8": (np.uint8, 1),
                        # 12/16-bit Bayer (uint16으로 처리)
                        "BayerRG12": (np.uint16, 1),
                        "BayerGB12": (np.uint16, 1),
                        "BayerGR12": (np.uint16, 1),
                        "BayerBG12": (np.uint16, 1),
                        "BayerRG16": (np.uint16, 1),
                    }
                    # 맵에 없는 포맷은 Mono8로 기본 처리
                    dt, ch = pf_map.get(pix_fmt, (np.uint8, 1))

                    # 4. 원시 메모리 버퍼로부터 NumPy 배열 생성 (복사본 생성)
                    view = cast(ptr, POINTER(c_ubyte * sz)).contents
                    data = np.frombuffer(view, dtype=dt).copy()

                    # 5. 1D 데이터를 2D 또는 3D 이미지 배열로 Reshape
                    frame = None
                    if ch == 3:  # 컬러 이미지 (RGB)
                        target_shape = (h, w, 3)
                        if data.size == h * w * 3:
                            frame = data.reshape(target_shape)
                            if pix_fmt == "BGR8":
                                frame = frame[..., ::-1]
                        else:
                            raise FrameAcquisitionError(
                                f"RGB Reshape mismatch! Data size {data.size} != "
                                f"Expected size {h * w * 3} for shape {target_shape} (format: {pix_fmt})"
                            )
                    else:  # 흑백 이미지 (Mono, Bayer)
                        target_shape = (h, w)
                        if data.size == h * w:
                            frame = data.reshape(target_shape)
                        else:
                            raise FrameAcquisitionError(
                                f"Mono/Bayer Reshape mismatch! Data size {data.size} != "
                                f"Expected size {h * w} for shape {target_shape} (format: {pix_fmt})"
                            )

                    # 6. 성공적으로 디코딩된 NumPy 프레임 캐시
                    self.set_last_np_frame(frame)

                    # 7. 통계 업데이트 및 시그널 발생
                    self.stats["frame_count"] += 1
                    self.stats["last_frame_time"] = time.time()
                    self._update_stats()

                    with suppress(Exception):
                        self.reshaped_frame_ready.emit(frame)
                        self.frame_ready.emit(frame)

                    return frame

        except TimeoutException:
            if count_timeout_error:
                self.stats["error_count"] += 1
                self._update_stats()
                self.logger.warning("[%s] Frame timeout after %d ms.", self.cam_id, timeout_ms)
            return None

        except GenTLException as e:
            raise FrameAcquisitionError(f"[{self.cam_id}] GenTL error during frame acquisition: {e}") from e

        except Exception as e:
            raise FrameAcquisitionError(f"[{self.cam_id}] Unexpected error processing frame: {e}") from e

    # --- END   OF PATCH src/core/camera_controller.py ----------------------------

    def run_synchronized_trigger_test(
            self,
            expected_triggers: int,
            poll_interval: float = 2.0,
    ):
        """
        멀티-카메라 HW-Trigger 동기화 장기 시험.
        (1) 모든 컨트롤러 CIC 카운터 리셋
        (2) Leader 가 StartCycle -or- 대체 명령 반복 실행
        (3) 누락 트리거 감시
        """
        from core.camera_exceptions import GrabberError, CommandExecutionError

        self.logger.info("[%s] Synchronized trigger test → %d cycles", self.cam_id, expected_triggers)

        # ① 풀에서 연결된 컨트롤러 수집
        from core import controller_pool
        ctrls = [c for c in controller_pool.controllers.values() if c.is_connected()]
        if not ctrls:
            raise GrabberError("No connected controllers found.")

        # ② CIC 카운터 reset
        for c in ctrls:
            dev = getattr(c.grabber, "device", None)
            if dev and "CycleLostTriggerCountReset" in dev.features():
                with suppress(Exception):
                    dev.execute("CycleLostTriggerCountReset")
            self.logger.debug("[%s] CIC reset done", c.cam_id)

        # ③ Leader 의 grabber.device 확보
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            # 리더가 device 모듈을 못 가진 경우 → 첫 번째로 가진 컨트롤러를 리더로 교체
            for c in ctrls:
                dev = getattr(c.grabber, "device", None)
                if dev:
                    self.logger.info("[%s] Delegating leader role to %s", self.cam_id, c.cam_id)
                    break
        if dev is None:
            raise CommandExecutionError("No grabber.device owns a Cycle generator.")

        # ④ StartCycle 계열 명령 후보
        start_cmds = [n for n in ("StartCycle", "CycleGeneratorStart", "Start") if n in dev.features()]
        if not start_cmds:
            raise CommandExecutionError("Neither StartCycle nor compatible command found in Device-module.")

        trigger_count = 0
        last_log = time.time()

        while trigger_count < expected_triggers:
            # 1) 트리거 전송 (첫 번째 성공하는 명령 사용)
            for cmd in start_cmds:
                try:
                    dev.execute(cmd)
                    break
                except Exception:
                    continue
            else:
                raise CommandExecutionError("Failed to execute any cycle-start command.")

            trigger_count += 1
            time.sleep(0.05)  # 카메라 수신 여유

            # 2) 주기적 상태 확인
            if time.time() - last_log >= poll_interval or trigger_count == expected_triggers:
                total_lost = 0
                status_msg = []
                for c in ctrls:
                    lost = 0
                    dev_c = getattr(c.grabber, "device", None)
                    if dev_c and "CycleLostTriggerCount" in dev_c.features():
                        lost = int(dev_c.get("CycleLostTriggerCount"))
                    total_lost += lost
                    status_msg.append(f"{c.cam_id}: lost={lost}")

                self.logger.info("Progress %d/%d – %s", trigger_count, expected_triggers, "; ".join(status_msg))
                if total_lost:
                    raise GrabberError(f"Trigger lost! total={total_lost}")
                last_log = time.time()

        self.logger.info("Synchronized trigger test PASSED – %d cycles, 0 lost", expected_triggers)

    # --- END   OF PATCH camera_controller.py --------------------------------------

    # ────────────────────────────── Statistics ───────────────────────────
    def _update_stats(self) -> None:
        now = time.time()
        if now - self._last_stats_update < self._stats_interval:
            return
        start = self.stats["start_time"]
        if start > 0:
            elapsed = now - start
            self.stats["fps"] = self.stats["frame_count"] / elapsed if elapsed else 0.0
            total = self.stats["frame_count"] + self.stats["error_count"]
            self.stats["frame_loss"] = self.stats["error_count"] / total if total else 0.0
        self._last_stats_update = now
        self.logger.debug("[%s] Stats: fps=%.2f loss=%.2f%%",
                          self.cam_id, self.stats["fps"],
                          self.stats["frame_loss"] * 100)

    def export_stats(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export camera statistics as JSON or return as dict.
        """
        with self._lock:  # Thread-safe
            self._update_stats()
            stats_data = {
                "cam_id": self.cam_id,
                "serial_number": self.serial_number,
                "timestamp": datetime.utcnow().isoformat(),
                "connected": self.connected,
                "grabbing": self.grabbing,
                "stats": self.stats.copy()
            }
            if output_path:
                try:
                    with open(output_path, "w") as f:
                        json.dump(stats_data, f, indent=2)
                    self.logger.info(f"[{self.cam_id}] Stats exported to {output_path}")
                except Exception as e:
                    self.logger.error(f"[{self.cam_id}] Failed to export stats: {e}")
            return stats_data

    # ------------------------------------------------------------------
    #  노드맵 탐색 유틸 – camera-remote → interface → device → remote → stream → system
    # ------------------------------------------------------------------
    def _find_nodemap(self, node_name: str):
        """
        Camera-remote(self.params)를 시작으로

            Interface → Device → Remote → Stream → System

        순서로 돌며 `features()` 안에 *node_name* 이 있는 첫 nodemap을
        돌려준다.  없으면 **None**.
        """
        if not self.connected:
            return None

        nm_candidates = [self.params]

        # Grabber가 제공하는 여러 nodemap 속성
        if self.grabber:
            for attr in ("interface", "device", "remote", "stream", "system"):
                nm = getattr(self.grabber, attr, None)
                if nm and nm not in nm_candidates:
                    nm_candidates.append(nm)

        for nm in nm_candidates:
            try:
                if node_name in nm.features():
                    return nm
            except Exception:
                # 일부 nodemap 은 features() 를 안 가질 수도 있다
                continue
        return None



    def set(self, name: str, value: Any, *, verify: bool = True) -> None:
        if not self.connected:
            raise CameraNotConnectedError(f"[{self.cam_id}] not connected")

        node_maps = [self.params, getattr(self.grabber, "remote", None)]

        def _write(nm):
            if hasattr(nm, "getEnum") and isinstance(value, str):
                try:
                    if value not in nm.getEnum(name):
                        raise ParameterError(f"{name} unsupported '{value}' (enum)")
                except Exception:
                    pass
            nm.set(name, value)
            if verify:
                with suppress(Exception):
                    if str(nm.get(name)) != str(value):
                        raise ParameterError(f"{name} verify failed")

        for nm in node_maps:
            if nm and name in nm.features():
                _write(nm)
                self.logger.info("[%s] %s ← %s (%s)",
                                 self.cam_id, name, value,
                                 "camera" if nm is self.params else "grabber")
                return

        # Legacy SFNC-map 보정
        for alt_name, alt_val in LEGACY_MAP.get(name, []):
            for nm in node_maps:
                if nm and alt_name in nm.features():
                    nm.set(alt_name, alt_val)
                    self.logger.debug("[%s] LEGACY_MAP %s → %s=%s",
                                      self.cam_id, name, alt_name, alt_val)
                    return
        raise ParameterError(f"{name} not available in camera/grabber nodemap")

    def grab_single_frame(self, timeout_ms: int = 1000) -> np.ndarray:  # noqa: D401
        """
        SequenceRunner·액션 모듈 호환용 thin-wrapper.
        Live-view 신호(`frame_ready`)도 그대로 전파됩니다.
        """
        return self.get_next_frame(timeout_ms=timeout_ms)

    def configure_grabber_cycle_trigger(
            self, *, trigger_source: str = "Immediate",
            cycle_period_us: float = 3.36, dlt_index: int = 1
    ) -> None:
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            raise AttributeError("Device-module (grabber.device) unavailable for cycle-trigger.")

        feats = dev.features()
        required_nodes_cycle = {"CycleGeneratorSelector", "CycleTriggerSource", "CycleTriggerActivation",
                                "CyclePeriodUs"}
        if not required_nodes_cycle.issubset(feats):
            # [수정] 경고 대신 예외 발생
            raise ParameterError(f"CycleGenerator related nodes are missing from grabber.device.")

        dev.set("CycleGeneratorSelector", 0)
        dev.set("CycleTriggerSource", trigger_source)
        dev.set("CycleTriggerActivation", "RisingEdge")
        dev.set("CyclePeriodUs", float(cycle_period_us))

        required_nodes_dlt = {"DeviceLinkTriggerToolSelector", "DeviceLinkTriggerToolSource"}
        if not required_nodes_dlt.issubset(feats):
            # [수정] 경고 대신 예외 발생
            raise ParameterError(f"DeviceLinkTriggerTool related nodes are missing from grabber.device.")

        dev.set("DeviceLinkTriggerToolSelector", int(dlt_index))
        dev.set("DeviceLinkTriggerToolSource", "CycleGenerator0")

        with suppress(Exception):
            if "StartCycle" in feats:
                dev.execute("StartCycle")
        self.logger.info("[%s] Cycle-trigger armed  src=%s  period=%.3f µs  DLT=%d",
                         self.cam_id, trigger_source, cycle_period_us, dlt_index)

    # ------------------------------------------------------------------ #
    #  Parameter setter (액션·API가 호출하는 공개 메서드)
    # ------------------------------------------------------------------ #
    def set_param(self, node_name: str, value: Any) -> None:
        """
        Public wrapper for setting parameters with safety checks.
        """
        with self._lock:
            self._check_connected()

            # For critical parameters that affect image size/format,
            # we must stop the grab if not in a persistent live-view session.
            critical_params = {"width", "height", "offsetx", "offsety", "pixelformat", "binninghorizontal", "binningvertical"}
            if node_name.lower() in critical_params and self.is_grabbing() and not self._live_view:
                self.logger.info("[%s] [Safety] Stopping grab before changing '%s'", self.cam_id, node_name)
                try:
                    self.stop_grab(flush=True)
                    time.sleep(0.1)
                except Exception as e:
                    self.logger.warning("[%s] Failed to stop grab for parameter change: %s", self.cam_id, e)

            self.set(node_name, value, verify=False)

            if self.enable_param_cache:
                self._param_cache[node_name] = value
                self._param_cache_timestamp[node_name] = time.time()

            self.logger.info("[%s] Param '%s' ← %s", self.cam_id, node_name, value)
    # ------------------------------------------------------------------
    #  안전한 파라미터 읽기
    # ------------------------------------------------------------------
    def get_param(self, node_name: str, force_reload: bool = False):
        """
        • `_find_nodemap` 으로 실제 노드 위치를 찾아 읽는다.
        • 캐시 사용 여부/갱신 로직은 기존과 동일.
        """
        with self._lock:
            self._check_connected()

            # ── 캐시 확인 ───────────────────────────────────────────
            if self.enable_param_cache and not force_reload:
                cached = self._param_cache.get(node_name)
                age    = time.time() - self._param_cache_timestamp.get(node_name, 0)
                if cached is not None and age < self.cache_timeout_sec:
                    return cached

            # ── 실제 nodemap 검색 ─────────────────────────────────
            nm = self._find_nodemap(node_name)
            if nm is None:
                raise ParameterError(f"[{self.cam_id}] Node '{node_name}' not found in any nodemap")

            try:
                val = nm.get(node_name)

                if self.enable_param_cache:
                    self._param_cache[node_name] = val
                    self._param_cache_timestamp[node_name] = time.time()

                return val

            except (GenTLException, TypeError) as e:
                raise ParameterError(
                    f"[{self.cam_id}] Failed to read '{node_name}': {e}"
                ) from e

    def get_device_vendor_name(self) -> str:
        with self._lock:
            try:
                # 수정: getattr로 get_info 안전하게 접근
                vendor = getattr(self.grabber, "get_info", lambda key: None)("DeviceVendorName")
                self.logger.info(f"[{self.cam_id}] Device vendor: {vendor}")
                return vendor if vendor is not None else "N/A"
            except Exception as e:
                self.logger.warning(f"[{self.cam_id}] Failed to get device vendor name: {e}")
                return "N/A"

    def get_device_model_name(self) -> str:
        with self._lock:
            try:
                # 수정: getattr로 get_info 안전하게 접근
                model = getattr(self.grabber, "get_info", lambda key: None)("DeviceModelName")
                self.logger.info(f"[{self.cam_id}] Device model: {model}")
                return model if model is not None else "N/A"
            except Exception as e:
                self.logger.warning(f"[{self.cam_id}] Failed to get device model name: {e}")
                return "N/A"

    def get_device_serial_number(self) -> str:
        with self._lock:
            # ① Grabber info → ② 원격 노드 순으로 시도
            serial = getattr(self.grabber, "get_info", lambda k: None)("DeviceSerialNumber")
            if not serial and self.params:
                try:
                    serial = self.params.get("DeviceSerialNumber")
                except Exception:
                    pass
            return serial or "N/A"

    # ────────────────────────────── 수정 ①
    def is_connected(self) -> bool:
        """스레드 안전 + 즉시 값 반환(락 불필요)."""
        return self.connected

    def _check_connected(self) -> None:
        if not self.connected:
            raise CameraNotConnectedError(f"[{self.cam_id}] Camera not connected.")

    def list_all_features(self, only_available: bool = True) -> List[str]:
        with self._lock:  # Thread-safe
            self._check_connected()
            if not self.params:
                self.logger.warning(f"[{self.cam_id}] No params object => cannot list features")
                return []
            try:
                feats = self.params.features(available_only=only_available)
                return feats
            except GenTLException as e:
                self.logger.error(f"[{self.cam_id}] Failed to list features: {e}", exc_info=True)
                return []
            except Exception as e:
                self.logger.error(f"[{self.cam_id}] Unexpected error listing features: {e}", exc_info=True)
                return []

    def get_enumeration_entries(self, node_name: str) -> List[str]:
        with self._lock:  # Thread-safe
            self._check_connected()
            if not self.params or not query:
                self.logger.warning(f"[{self.cam_id}] No params or query module => cannot get enum for {node_name}")
                return []
            try:
                q_enum = query.enum_entries(node_name, available_only=True)
                entries = self.params.get(q_enum, list)
                self.logger.debug(f"[{self.cam_id}] [ENUM] {node_name} => {entries}")
                return entries if entries is not None else []
            except GenTLException as e:
                self.logger.debug(f"[{self.cam_id}] get_enumeration_entries({node_name}) failed: {e}")
                return []
            except Exception as e:
                self.logger.warning(f"[{self.cam_id}] Unexpected error in get_enumeration_entries({node_name}): {e}")
                return []

    # ──────────────────────────────────────────────
    # Device-module 파라미터 읽기 / 쓰기 / Command
    # ──────────────────────────────────────────────
    def set_device_param(self, node_name: str, value) -> None:
        """
        Grabber(Device-module) GenICam 노드에 값을 쓰는 헬퍼.
        """
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            raise AttributeError("grabber.device is None (Device-module not supported?)")

        if node_name not in dev.features():
            raise ParameterError(f"Node '{node_name}' not found in Device-module.")

        dev.set(node_name, value)
        self.logger.info(f"[{self.cam_id}] [Device] {node_name} ← {value}")

    def get_device_param(self, node_name: str):
        """
        Grabber(Device-module) 노드 값을 읽는 헬퍼.
        """
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            raise AttributeError("grabber.device is None (Device-module not supported?)")

        if node_name not in dev.features():
            raise ParameterError(f"Node '{node_name}' not found in Device-module.")

        val = dev.get(node_name)
        self.logger.debug(f"[{self.cam_id}] [Device] {node_name} ⇒ {val}")
        return val

    def trigger_software_safe(self, *, cxp_id: int = 0) -> None:
        self._check_connected()
        original = self.params.get("TriggerSource")
        try:
            if original != "Software":
                self.params.set("TriggerSource", "Software")
            self.execute_software_trigger(cxp_id=cxp_id)
            self.logger.info("[%s] Software trigger sent (safe)", self.cam_id)
        finally:
            if original != "Software":
                with suppress(Exception):
                    self.params.set("TriggerSource", original)

    def execute_device_command(self, node_name: str, timeout: float = 0.5) -> None:
        """
        Device-module에 존재하는 *Command* 노드를 실행한다.
        """
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            raise AttributeError("grabber.device is None (Device-module not supported?)")

        node = dev.getNode(node_name)
        if not (node and node.isValid() and node.isWritable()):
            raise CommandExecutionError(f"Device command '{node_name}' not found or not writable.")

        node.execute()
        if timeout > 0 and hasattr(node, 'wait_until_done'):
            node.wait_until_done(timeout)

        self.logger.info(f"[{self.cam_id}] Device-command '{node_name}' executed.")

    def is_writeable(self, node_name: str) -> bool:
        with self._lock:  # Thread-safe
            self._check_connected()
            if not self.params or not query:
                self.logger.debug(f"[{self.cam_id}] Cannot check writeable for {node_name}: No params or query")
                return False
            try:
                q_ = query.writeable(node_name)
                return bool(self.params.get(q_, bool))
            except GenTLException as e:
                self.logger.debug(f"[{self.cam_id}] is_writeable({node_name}) failed: {e}")
                return False
            except Exception as e:
                self.logger.warning(f"[{self.cam_id}] Unexpected error in is_writeable({node_name}): {e}")
                return False

    def get_parameter_metadata(self, name: str) -> Dict[str, Any]:
        with self._lock:  # Thread-safe
            self._check_connected()
            if not self.params:
                raise ParameterError(f"[{self.cam_id}] Camera parameters not initialized.")
            metadata = {}
            try:
                value = self.params.get(name)
                metadata["value"] = value
                self.logger.debug(f"[{self.cam_id}] Metadata for {name}: value={value}")
            except Exception as e:
                metadata["value"] = None
                metadata["error"] = f"Failed to get value: {str(e)}"
                self.logger.warning(f"[{self.cam_id}] Failed to get value for {name}: {e}")
                return metadata
            try:
                metadata["is_writeable"] = self.is_writeable(name)
                metadata["access"] = "RW" if metadata["is_writeable"] else "RO"
                self.logger.debug(f"[{self.cam_id}] {name} access={metadata['access']}")
                metadata["type"] = self._guess_type(name)
                self.logger.debug(f"[{self.cam_id}] Guessed type for {name}: {metadata['type']}")
                for key in ["Min", "Max", "Inc"]:
                    try:
                        q_info = query.info(name, key)
                        metadata[key.lower()] = self.params.get(q_info, type(value) if key != "Inc" else float)
                        self.logger.debug(f"[{self.cam_id}] {name} {key}={metadata[key.lower()]}")
                    except Exception as e:
                        metadata[key.lower()] = None
                        self.logger.debug(f"[{self.cam_id}] No {key} for {name}: {e}")
                        if key == "Min" and metadata["type"] in ["Integer", "Float"]:
                            metadata[key.lower()] = 0
                        elif key == "Max" and metadata["type"] == "Integer":
                            metadata[key.lower()] = 2 ** 31 - 1
                        elif key == "Max" and metadata["type"] == "Float":
                            metadata[key.lower()] = 1.0e18
                        elif key == "Inc":
                            metadata[key.lower()] = 1 if metadata["type"] == "Integer" else 0.1
                try:
                    q_unit = query.info(name, "Unit")
                    metadata["unit"] = self.params.get(q_unit, str) if q_unit else ""
                    self.logger.debug(f"[{self.cam_id}] {name} Unit={metadata['unit']}")
                except Exception as e:
                    metadata["unit"] = ""
                    self.logger.debug(f"[{self.cam_id}] No Unit for {name}: {e}")
                    if name.lower() in ["width", "height", "offsetx", "offsety"]:
                        metadata["unit"] = "px"
                    elif "rate" in name.lower() or "frequency" in name.lower():
                        metadata["unit"] = "fps"
                    elif "time" in name.lower():
                        metadata["unit"] = "us"
                try:
                    q_desc = query.info(name, "Description") or query.info(name, "DisplayName") or query.info(name, "ToolTip")
                    metadata["description"] = self.params.get(q_desc, str) if q_desc else name
                    self.logger.debug(f"[{self.cam_id}] {name} Description={metadata['description']}")
                except Exception as e:
                    metadata["description"] = name
                    self.logger.debug(f"[{self.cam_id}] No Description for {name}: {e}")
                if metadata["type"] == "Enumeration":
                    try:
                        enum_entries = self.get_enumeration_entries(name)
                        metadata["enum_entries"] = enum_entries
                        self.logger.debug(f"[{self.cam_id}] {name} Enum Entries={enum_entries}")
                    except Exception as e:
                        metadata["enum_entries"] = []
                        metadata["enum_error"] = f"Failed to get enum entries: {e}"
                        self.logger.warning(f"[{self.cam_id}] Failed to get enum entries for {name}: {e}")
            except Exception as e:
                self.logger.error(f"[{self.cam_id}] Metadata retrieval failed for {name}: {e}", exc_info=True)
            return metadata

    def _guess_type(self, name: str) -> str:
        with self._lock:  # Thread-safe
            try:
                value = self.get_param(name)
                if value is None:
                    return 'String'
                if isinstance(value, bool):
                    return 'Boolean'
                if isinstance(value, int):
                    return 'Integer'
                if isinstance(value, float):
                    return 'Float'
                if isinstance(value, str):
                    try:
                        enum_entries = self.get_enumeration_entries(name)
                        if enum_entries and len(enum_entries) > 0:
                            self.logger.debug(f"[{self.cam_id}] {name} detected as Enumeration with entries: {enum_entries}")
                            return 'Enumeration'
                    except Exception as e:
                        self.logger.debug(f"[{self.cam_id}] No enum entries for {name}: {e}")
                    return 'String'
                return 'String'
            except Exception as e:
                self.logger.warning(f"[{self.cam_id}] Failed to guess type for {name}: {e}")
                return 'String'

    def set_exposure(self, exposure_us: float) -> None:
        with self._lock:
            self.set_param("ExposureTime", float(exposure_us))
            # 수정: f-string에서 'us'를 문자열로 포함
            self.logger.info(f"[{self.cam_id}] Exposure set to {exposure_us} us")

    def get_exposure(self) -> float:
        with self._lock:  # Thread-safe
            return float(self.get_param("ExposureTime"))

    def set_gain(self, gain: float) -> None:
        with self._lock:  # Thread-safe
            self.set_param("Gain", float(gain))
            self.logger.info(f"[{self.cam_id}] Gain set to {gain}")

    def get_gain(self) -> float:
        with self._lock:  # Thread-safe
            return float(self.get_param("Gain"))

    # src/core/camera_controller.py
    # ─────────────────────────────────────────────────────────
    def execute_software_trigger(self, *, cxp_id: int = 0) -> None:
        """
        ① 카메라/Grabber remote 의 *TriggerSoftware* 가 있으면 우선 사용
        ② 없으면 Device-Module 의 **CxpTriggerMessageSend** 로 폴백
           (필요 시 CxpTriggerMessageID 선행 세팅)
        """
        # ── ① 가장 일반적인 TriggerSoftware ──────────────────────
        try:
            self.execute_command("TriggerSoftware", timeout=0.5)
            return
        except CommandExecutionError:
            self.logger.debug("[%s] TriggerSoftware unavailable – trying CXP message",
                              self.cam_id)

        # ── ② CXP Trigger Message (grabber.device) ───────────────
        dev = getattr(self.grabber, "device", None)
        if dev is None:
            raise CommandExecutionError(f"[{self.cam_id}] No DeviceModule for CXP trigger")

        try:
            if "CxpTriggerMessageID" in dev.features():
                dev.set("CxpTriggerMessageID", cxp_id)
            self.execute_command("CxpTriggerMessageSend", timeout=0.5)
            self.logger.info("[%s] CxpTriggerMessageSend(ID=%d) OK", self.cam_id, cxp_id)
        except Exception as exc:
            raise CommandExecutionError(
                f"[{self.cam_id}] Failed to send CXP trigger: {exc}"
            ) from exc

