# core/controller_pool.py

# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
controller_pool.py  ·  Build 2025-06-20-PATCH
──────────────────────────────────────────────────────────────────────────────
Thread-safe **global registry** for multiple ``CameraController`` instances.
순환 참조 문제를 해결하기 위해 TYPE_CHECKING을 사용합니다.
"""
from __future__ import annotations
import time
import logging
import threading
from typing import Dict, Any, Callable, Iterable, List, Optional, Union, TYPE_CHECKING

# --- 순환 참조 해결 ---
if TYPE_CHECKING:
    from .camera_controller import CameraController

__all__ = [
    "register", "unregister", "flush", "get", "get_controller",
    "broadcast", "list_ids", "ids", "controllers", "first_controller",
    "connect_all"
]

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
#  Global registry  (cam_id → active CameraController)
# ────────────────────────────────────────────────────────────────────────────
controllers: Dict[str, "CameraController"] = {}
_lock = threading.RLock()


# ---------------------------------------------------------------------------
#  Low-level helpers
# ---------------------------------------------------------------------------
def _next_unique_id(base: str) -> str:
    """Return ``base`` or ``base_1, base_2, …`` so that it is unique inside
    the *current* registry (thread-safe)."""
    with _lock:
        if base not in controllers:
            return base
        suffix = 1
        while f"{base}_{suffix}" in controllers:
            suffix += 1
        return f"{base}_{suffix}"


# ---------------------------------------------------------------------------
#  Public-API helpers (module-level)
# ---------------------------------------------------------------------------
def get_controller(cam_id: str) -> Optional["CameraController"]:
    """
    Back-compat alias for :pyfunc:`get`.
    """
    return get(cam_id)


# ──────────────────────────────────────────────────────────────────────────
#  REGISTER / UNREGISTER
# ──────────────────────────────────────────────────────────────────────────
def register(controller: "CameraController", *, allow_replace: bool = False) -> str:
    """
    Add *controller* to the global pool.
    """
    cid_base = controller.cam_id or "UNKNOWN"
    with _lock:
        if cid_base in controllers:
            if allow_replace or not controllers[cid_base].is_connected():
                logger.warning(f"[Pool] '{cid_base}' 을(를) 기존 항목과 교체합니다.")
                controllers[cid_base] = controller
                cid_final = cid_base
            else:
                cid_final = _next_unique_id(cid_base)
                controller.cam_id = cid_final
                controllers[cid_final] = controller
                logger.warning(f"[Pool] 중복 cam_id '{cid_base}' → '{cid_final}' 로 저장.")
        else:
            controllers[cid_base] = controller
            cid_final = cid_base
            logger.debug(f"[Pool] 새 컨트롤러 등록: {cid_final}")
    return cid_final


def unregister(cam_id: str) -> None:
    with _lock:
        if cam_id in controllers:
            controllers.pop(cam_id, None)
            logger.debug(f"[Pool] 컨트롤러 등록 해제: {cam_id}")


def flush(*, disconnect: bool = True, grace_ms: int = 300) -> None:
    with _lock:
        items = list(controllers.items())
        controllers.clear()

    if disconnect:
        logger.info(f"[Pool] {len(items)}개의 컨트롤러 연결 해제 및 풀 비우기 시작...")
        for cid, ctrl in items:
            try:
                ctrl.disconnect_camera()
            except Exception as e:
                logger.warning(f"[Pool] flush 중 {cid} 연결 해제 실패: {e}")
    if grace_ms > 0:
        time.sleep(grace_ms / 1000.0)
    logger.info(f"[Pool] 풀 비우기 완료 (grace {grace_ms}ms)")


# ──────────────────────────────────────────────────────────────────────────
#  SIMPLE QUERIES
# ──────────────────────────────────────────────────────────────────────────
def first_controller() -> Optional["CameraController"]:
    """연결된 컨트롤러 중 첫 번째 항목을 반환합니다."""
    with _lock:
        sorted_ids = sorted(controllers.keys())
        for cid in sorted_ids:
            ctrl = controllers[cid]
            if ctrl.is_connected():
                return ctrl
        return None


def get(cam_id: str) -> Optional["CameraController"]:
    """스레드 안전한 조회 헬퍼."""
    with _lock:
        return controllers.get(cam_id)


def list_ids() -> List[str]:
    """현재 cam_id 목록의 스냅샷을 반환합니다 (정렬됨)."""
    with _lock:
        return sorted(controllers.keys())


def ids() -> List[str]:
    """
    **현재 연결(connected=True) 상태**인 컨트롤러들의 cam_id 리스트를 반환합니다.
    """
    with _lock:
        return sorted([cid for cid, ctrl in controllers.items() if ctrl.is_connected()])


# ---------------------------------------------------------------------------
#  connect_all – discover + connect every detectable camera
# ---------------------------------------------------------------------------
def connect_all(*, enable_param_cache: bool = False, replace_existing: bool = False) -> List[str]:
    """Discover, connect and *register* every camera in the system."""
    # `from .camera_controller import CameraController`가 필요하므로 함수 내부에서 임포트합니다.
    # 이것은 순환 참조를 피하는 또 다른 방법입니다.
    from .camera_controller import CameraController

    newly: List[str] = []
    discovered = CameraController.connect_all(enable_param_cache=enable_param_cache)
    for ctrl in discovered:
        cid = register(ctrl, allow_replace=replace_existing)
        newly.append(cid)
    logger.info("connect_all(): new=%d  total=%d", len(newly), len(controllers))
    return newly


# ---------------------------------------------------------------------------
#  broadcast – invoke same method on every controller
# ---------------------------------------------------------------------------
def broadcast(method: str, *args: Any, **kwargs: Any) -> Dict[str, Union[Any, Exception]]:
    with _lock:
        snapshot = list(controllers.items())
    results: Dict[str, Union[Any, Exception]] = {}
    silent = kwargs.pop('silent', False)
    for cid, ctrl in snapshot:
        try:
            fn: Callable = getattr(ctrl, method)
            results[cid] = fn(*args, **kwargs)
        except Exception as exc:                     # noqa: BLE001
            results[cid] = exc
            if not silent:
                logger.error(f"[{cid}] broadcast('{method}') 실패: {exc}", exc_info=True)
    return results

# ════════════════════════════════════════════════════════════════════════════
#  OO-style wrapper – optional convenience around same logic
# ════════════════════════════════════════════════════════════════════════════
# (변경 없음 – 원본 유지)
class ControllerPool:
    """Thread-safe registry managing multiple *connected* ``CameraController`` objects."""
    _GLOBAL: "ControllerPool | None" = None  # lazy singleton

    # ------------------------------------------------------------------ basic
    def __init__(self) -> None:
        self._controllers: Dict[str, CameraController] = {}
        self._lock = threading.RLock()

    def add(self, controller: CameraController, *, allow_replace: bool = False) -> str:
        cid_base = controller.cam_id or "UNKNOWN"
        with self._lock:
            if cid_base in self._controllers and not allow_replace:
                cid_final = _next_unique_id(cid_base)
                logger.warning("[LocalPool] duplicate '%s' → stored as '%s'",
                               cid_base, cid_final)
            else:
                cid_final = cid_base
            self._controllers[cid_final] = controller
        return cid_final

    def remove(self, cam_id: str) -> None:
        with self._lock:
            self._controllers.pop(cam_id, None)
        logger.info("[LocalPool] removed %s", cam_id)

    def get(self, cam_id: str) -> Optional[CameraController]:
        with self._lock:
            return self._controllers.get(cam_id)

    # iterable / sized behaviour ------------------------------------------------
    def __iter__(self) -> Iterable[CameraController]:
        with self._lock:
            return iter(list(self._controllers.values()))  # snapshot

    def __len__(self) -> int:                               # pragma: no cover
        with self._lock:
            return len(self._controllers)

    def list_ids(self) -> List[str]:
        with self._lock:
            return sorted(self._controllers.keys())

    # utilities -----------------------------------------------------------------
    def broadcast(self, cmd: str, *args: Any, raise_on_error: bool = False, **kwargs: Any) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        with self._lock:
            snapshot = list(self._controllers.items())
        for cam_id, ctrl in snapshot:
            try:
                fn: Callable = getattr(ctrl, cmd)
                results[cam_id] = fn(*args, **kwargs)
            except Exception as exc:                     # noqa: BLE001
                logger.exception("broadcast(%s) failed for %s", cmd, cam_id)
                if raise_on_error:
                    raise
                results[cam_id] = exc
        return results

    # stats ---------------------------------------------------------------------
    def stats(self) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            snapshot = list(self._controllers.items())
        for cam_id, ctrl in snapshot:
            try:
                stats[cam_id] = ctrl.export_stats()
            except Exception as exc:                     # noqa: BLE001
                logger.exception("stats() failed for %s", cam_id)
                stats[cam_id] = {"error": str(exc)}
        return stats

    # discovery -----------------------------------------------------------------
    @classmethod
    def global_instance(cls) -> "ControllerPool":
        if cls._GLOBAL is None:
            cls._GLOBAL = ControllerPool()
        return cls._GLOBAL

    @classmethod
    def connect_all(cls, *, enable_param_cache: bool = False, replace_existing: bool = False) -> List[str]:
        return connect_all(enable_param_cache=enable_param_cache,
                           replace_existing=replace_existing)
