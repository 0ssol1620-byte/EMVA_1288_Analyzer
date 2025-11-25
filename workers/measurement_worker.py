# workers/measurement_worker.py

from PyQt5.QtCore import QThread, pyqtSignal
import time
import numpy as np
from typing import Optional, Tuple
from contextlib import suppress

# 실제 프로젝트 경로에서 import
from core.camera_facade import CxpCamera
from core.lightbox_controller import LightBoxController

DEFAULT_LB_MAX_UNITS = 65535
VALID_BAYER = ["RGGB", "GRBG", "GBRG", "BGGR"]


# ---------------- EMVA 1288 helpers (PRNU/DSNU, shading 제거) ----------------

def _fit_plane_z(xy_img: np.ndarray) -> np.ndarray:
    if xy_img is None or xy_img.ndim != 2:
        return xy_img
    h, w = xy_img.shape
    y, x = np.mgrid[0:h, 0:w]
    X = np.column_stack([x.ravel(), y.ravel(), np.ones((h * w,), dtype=np.float64)])
    z = xy_img.astype(np.float64).ravel()
    coeff, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
    plane = (coeff[0] * x + coeff[1] * y + coeff[2]).reshape(h, w)
    return plane


def _remove_shading(image_2d: np.ndarray) -> np.ndarray:
    if image_2d is None or image_2d.ndim != 2:
        return image_2d
    plane = _fit_plane_z(image_2d)
    return image_2d.astype(np.float64) - plane


def _g_sites_from_bayer(raw: np.ndarray, pattern: Optional[str]) -> Optional[np.ndarray]:
    if raw is None or raw.ndim != 2 or pattern not in VALID_BAYER:
        return None
    H, W = raw.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    f = raw[:H2, :W2].astype(np.float64)
    p = pattern.upper()
    if p == "RGGB":
        sl_gr = (slice(0, None, 2), slice(1, None, 2))
        sl_gb = (slice(1, None, 2), slice(0, None, 2))
    elif p == "BGGR":
        sl_gr = (slice(1, None, 2), slice(0, None, 2))
        sl_gb = (slice(0, None, 2), slice(1, None, 2))
    elif p == "GRBG":
        sl_gr = (slice(0, None, 2), slice(0, None, 2))
        sl_gb = (slice(1, None, 2), slice(1, None, 2))
    elif p == "GBRG":
        sl_gr = (slice(1, None, 2), slice(1, None, 2))
        sl_gb = (slice(0, None, 2), slice(0, None, 2))
    else:
        return None
    return np.concatenate([f[sl_gr].ravel(), f[sl_gb].ravel()], axis=0)


def _estimate_dsnu_from_dark_pairs(raw_pairs: list,
                                   pattern: Optional[str],
                                   remove_shading_plane: bool = True) -> Optional[float]:
    if not raw_pairs:
        return None
    acc = None
    cnt = 0
    for fa, fb in raw_pairs:
        if fa is None or fb is None or fa.ndim != 2 or fb.ndim != 2:
            continue
        m = (fa.astype(np.float64) + fb.astype(np.float64)) * 0.5
        if acc is None:
            acc = m.copy()
        else:
            acc += m
        cnt += 1
    if acc is None or cnt == 0:
        return None
    avg = acc / float(cnt)

    if pattern in VALID_BAYER:
        H, W = avg.shape
        avg_c = avg[:(H // 2) * 2, :(W // 2) * 2]
        if remove_shading_plane:
            avg_c = _remove_shading(avg_c)
        gs = _g_sites_from_bayer(avg_c, pattern)
        if gs is None or gs.size == 0:
            return None
        return float(np.std(gs))
    else:
        f = avg.astype(np.float64)
        if remove_shading_plane:
            f = _remove_shading(f)
        f = f - f.mean()
        return float(np.sqrt(np.var(f)))


def _estimate_prnu_rel(fa: np.ndarray,
                       fb: np.ndarray,
                       pattern: Optional[str],
                       mu_dn: float,
                       sigma_temp_dn: float,
                       remove_shading_plane: bool = True) -> Optional[float]:
    if fa is None or fb is None or fa.shape != fb.shape or mu_dn <= 0:
        return None
    m = (fa.astype(np.float64) + fb.astype(np.float64)) * 0.5

    if pattern in VALID_BAYER and m.ndim == 2:
        H, W = m.shape
        mc = m[:(H // 2) * 2, :(W // 2) * 2]
        if remove_shading_plane:
            mc = _remove_shading(mc)
        gs = _g_sites_from_bayer(mc, pattern)
        if gs is None or gs.size == 0:
            return None
        s_res = float(np.std(gs))
    else:
        if m.ndim != 2:
            m = np.mean(m, axis=-1) if m.ndim == 3 else m
        f = _remove_shading(m) if remove_shading_plane else m.astype(np.float64)
        s_res = float(np.std(f))

    var_res = max(s_res * s_res - 0.5 * (sigma_temp_dn * sigma_temp_dn), 0.0)
    return float(np.sqrt(var_res) / float(mu_dn))


# ---------------------------------------------------------------------
# 유틸: Bayer → G-compact, RGB 언팩, G-site 기반 노이즈/분산 계산
# ---------------------------------------------------------------------
def _bayer_green_compact(raw: np.ndarray, pattern: Optional[str]) -> Optional[np.ndarray]:
    if raw is None or raw.ndim != 2 or pattern not in VALID_BAYER:
        return None
    H, W = raw.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    r = raw[:H2, :W2]
    p = pattern.upper()
    if p == "RGGB":
        Gr = r[0::2, 1::2]
        Gb = r[1::2, 0::2]
    elif p == "BGGR":
        Gr = r[1::2, 0::2]
        Gb = r[0::2, 1::2]
    elif p == "GRBG":
        Gr = r[0::2, 0::2]
        Gb = r[1::2, 1::2]
    elif p == "GBRG":
        Gr = r[1::2, 1::2]
        Gb = r[0::2, 0::2]
    else:
        return None
    return ((Gr.astype(np.float64) + Gb.astype(np.float64)) * 0.5).astype(r.dtype)


def _reshape_if_rgb_unpacked_worker(frame: np.ndarray, pixel_format: str, camera) -> Optional[np.ndarray]:
    if frame is None:
        return None
    pf_up = (pixel_format or "").upper()
    if ("RGB" in pf_up) and (frame.ndim != 3):
        try:
            h = int(camera.get("Height"))
            w = int(camera.get("Width"))
            if frame.dtype != np.uint16:
                frame = frame.view(np.uint16)
            return frame.reshape((h, w, 3))
        except Exception:
            return frame
    return frame


def _to_g_mono(frame: np.ndarray, pixel_format: str, pattern: Optional[str]) -> np.ndarray:
    if frame is None:
        return None
    pf = (pixel_format or "").upper()

    if frame.ndim == 2 and pattern in VALID_BAYER:
        gc = _bayer_green_compact(frame, pattern)
        return gc if gc is not None else frame

    if "RGB" in pf and frame.ndim == 3 and frame.shape[2] >= 2:
        return frame[..., 1]

    if frame.ndim == 2:
        return frame

    return frame[..., 1] if (frame.ndim == 3 and frame.shape[2] >= 2) else frame


def _std_temporal_green_mask(fa: np.ndarray, fb: np.ndarray, pattern: Optional[str]) -> Optional[float]:
    if fa is None or fb is None or fa.ndim != 2 or fb.ndim != 2 or pattern not in VALID_BAYER:
        return None
    H, W = fa.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    a = fa[:H2, :W2].astype(np.float64)
    b = fb[:H2, :W2].astype(np.float64)

    p = pattern.upper()
    if p == "RGGB":
        sl_gr = (slice(0, None, 2), slice(1, None, 2))
        sl_gb = (slice(1, None, 2), slice(0, None, 2))
    elif p == "BGGR":
        sl_gr = (slice(1, None, 2), slice(0, None, 2))
        sl_gb = (slice(0, None, 2), slice(1, None, 2))
    elif p == "GRBG":
        sl_gr = (slice(0, None, 2), slice(0, None, 2))
        sl_gb = (slice(1, None, 2), slice(1, None, 2))
    elif p == "GBRG":
        sl_gr = (slice(1, None, 2), slice(1, None, 2))
        sl_gb = (slice(0, None, 2), slice(0, None, 2))
    else:
        return None

    d = (a - b) / np.sqrt(2.0)
    g_gr = d[sl_gr]
    g_gb = d[sl_gb]
    return float(np.std(np.concatenate([g_gr.ravel(), g_gb.ravel()], axis=0)))


def _spatial_var_green_mask(frame: np.ndarray,
                            pattern: Optional[str],
                            remove_dc: bool = True,
                            ddof: int = 0) -> Optional[float]:
    if frame is None or frame.ndim != 2 or pattern not in VALID_BAYER:
        return None
    H, W = frame.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    f = frame[:H2, :W2].astype(np.float64)

    p = pattern.upper()
    if p == "RGGB":
        sl_gr = (slice(0, None, 2), slice(1, None, 2))
        sl_gb = (slice(1, None, 2), slice(0, None, 2))
    elif p == "BGGR":
        sl_gr = (slice(1, None, 2), slice(0, None, 2))
        sl_gb = (slice(0, None, 2), slice(1, None, 2))
    elif p == "GRBG":
        sl_gr = (slice(0, None, 2), slice(0, None, 2))
        sl_gb = (slice(1, None, 2), slice(1, None, 2))
    elif p == "GBRG":
        sl_gr = (slice(1, None, 2), slice(1, None, 2))
        sl_gb = (slice(0, None, 2), slice(0, None, 2))
    else:
        return None

    g = np.concatenate([f[sl_gr].ravel(), f[sl_gb].ravel()], axis=0)
    if remove_dc:
        g = g - g.mean()
    return float(np.var(g, ddof=ddof))


# ---------------------------------------------------------------------
# 공통 스냅 믹스인 (EMVA용 snap_pair – TDI 워밍업 포함)
# ---------------------------------------------------------------------
class _SnapMixin:
    def _reset_stream(self):
        try:
            if hasattr(self.camera, "stop_stream"):
                self.camera.stop_stream()
        except Exception:
            pass
        try:
            if hasattr(self.camera, "flush_stream"):
                self.camera.flush_stream()
        except Exception:
            pass
        try:
            if hasattr(self.camera, "start_stream"):
                self.camera.start_stream()
        except Exception:
            pass

    # ───────────────── TDI 정보 헬퍼 ─────────────────
    def _get_tdi_info(self) -> Tuple[bool, int]:
        """
        현재 카메라가 Linescan/TDI 인지와 스테이지 개수(없으면 0)를 반환.
        """
        is_tdi = False
        stages = 0

        with suppress(Exception):
            scan = str(self.camera.get("DeviceScanType") or "")
            if scan == "Linescan":
                is_tdi = True

        if not is_tdi:
            return False, 0

        with suppress(Exception):
            val = self.camera.get("TDIStages")
            if isinstance(val, str):
                digits = "".join(ch for ch in val if ch.isdigit())
                if digits:
                    stages = int(digits)
            elif isinstance(val, (int, float)):
                code = int(val)
                code_map = {
                    1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 96,
                    7: 128, 8: 160, 9: 192, 10: 224,
                    11: 240, 12: 248, 13: 252, 14: 256,
                }
                stages = code_map.get(code, 0)

        return True, max(0, stages)

    def _snap_pair_tdi(self, stages: int, delay_s: float):
        """
        Linescan/TDI 전용 snap_pair:
        - grab 을 시작하고,
        - warmup_frames = max(2*stages, 16) 만큼 프레임을 버린 뒤
        - 실제 사용할 두 프레임(fa, fb)을 읽어서 반환.
        우리가 시작한 grab 만 stop_grab(flush=False) 로 정리.
        """
        ctrl = self.camera.ctrl
        if ctrl is None:
            return None, None

        warmup_frames = max(2 * stages, 16) if stages > 0 else 16
        pre_started = False

        if not ctrl.is_grabbing():
            buf_count = max(8, warmup_frames + 4)
            ctrl.start_grab(buffer_count=buf_count)
            pre_started = True

        fa = fb = None
        try:
            # 워밍업: TDI 파이프라인 채우기
            for _ in range(warmup_frames):
                with suppress(Exception):
                    ctrl.get_next_frame(timeout_ms=1000)

            # 실제 측정용 프레임 두 장
            fa = ctrl.get_next_frame(timeout_ms=3000)
            if delay_s > 0:
                time.sleep(delay_s)
            fb = ctrl.get_next_frame(timeout_ms=3000)
        finally:
            if pre_started:
                with suppress(Exception):
                    ctrl.stop_grab(flush=False)

        return fa, fb

    def _snap_pair_safe(self, max_retries: int = 2, delay_s: float = 0.02):
        """
        EMVA 전용 snap_pair:
        - Area 카메라  → CxpCamera.snap_pair() 그대로 사용
        - Linescan/TDI → 내부에서 TDI 워밍업 후 두 프레임을 반환
        - GenTL/timeout 등 에러 발생 시 TL 리셋 후 재시도
        """
        last_err = None

        for _ in range(max_retries + 1):
            try:
                is_tdi, stages = self._get_tdi_info()
                if is_tdi:
                    fa, fb = self._snap_pair_tdi(stages, delay_s)
                else:
                    # 기존 area 카메라 경로
                    fa, fb = self.camera.snap_pair(delay_ms=int(delay_s * 1000))

                if fa is None or fb is None:
                    raise RuntimeError("snap_pair returned None")

                return fa, fb

            except Exception as e:
                last_err = e
                msg = str(e)
                needs_reset = (
                    "GenTL" in msg
                    or "GC_ERR_BUSY" in msg
                    or "-1" in msg
                    or "timeout" in msg.lower()
                    or "payload" in msg.lower()
                )
                time.sleep(delay_s)
                if needs_reset:
                    self._reset_stream()
                    time.sleep(delay_s)
                    continue
                else:
                    break

        raise RuntimeError(f"Safe snap failed after retries: {last_err}")

    def _to_lb_units(self, app_level: int) -> int:
        app = max(0, min(65535, int(app_level)))
        if self.lb_max_units >= 65535:
            return app
        return int(round(app * self.lb_max_units / 65535.0))


class SaturationWorker(QThread, _SnapMixin):
    progress_updated = pyqtSignal(int, str)
    saturation_found = pyqtSignal(int)  # clip level (app scale 0~65535)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    preview_ready = pyqtSignal(object)

    def __init__(self,
                 camera: CxpCamera,
                 lightbox: LightBoxController,
                 pixel_format: str,
                 bayer_pattern: Optional[str] = None,
                 lb_max_units: int = DEFAULT_LB_MAX_UNITS):
        super().__init__()
        self.camera = camera
        self.lightbox = lightbox
        self.pixel_format = pixel_format or ""
        self.bayer_pattern = bayer_pattern if (bayer_pattern in VALID_BAYER) else None
        self.lb_max_units = int(lb_max_units)
        self._stop = False

        # ROI / 검색 파라미터
        self.roi_ratio = 0.5
        self.rstride = 1
        self.cstride = 1
        self.top_percent = 0.001
        self.slack = 2
        self.preview_every = 2

        # DN이 full-scale의 몇 % 이상이면 "포화 근처"로 보고
        # 2배수(exponential) 성장을 멈출지 비율
        self.exp_stop_ratio = 0.75  # 예: 0.75 → full-scale의 75% 넘으면 포화 근처

        # ---- Linescan / TDI 여부 및 스테이지 수 ----
        self.is_linescan = False
        self._tdi_stages = 0
        with suppress(Exception):
            if str(self.camera.get("DeviceScanType") or "") == "Linescan":
                self.is_linescan = True
        if self.is_linescan:
            with suppress(Exception):
                val = self.camera.get("TDIStages")
                if isinstance(val, str):
                    digits = "".join(ch for ch in val if ch.isdigit())
                    if digits:
                        self._tdi_stages = int(digits)
                elif isinstance(val, (int, float)):
                    code_map = {
                        1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 96,
                        7: 128, 8: 160, 9: 192, 10: 224,
                        11: 240, 12: 248, 13: 252, 14: 256,
                    }
                    self._tdi_stages = code_map.get(int(val), 0)

        # 한 번만 버리는 초기 워밍업, 라이트 레벨별 워밍업
        # MeasurementWorker(_snap_pair_tdi)와 동일하게 2×stages로 통일
        if self.is_linescan and self._tdi_stages > 1:
            self._initial_warmup = max(2 * self._tdi_stages, 32)   # 시작 시
            self._per_level_warmup = max(2 * self._tdi_stages, 32) # 레벨 변경 시
        else:
            self._initial_warmup = 8
            self._per_level_warmup = 4

    def stop(self):
        self._stop = True

    # ------------------------------------------------------------------ helpers
    def _max_dn_from_pixfmt(self) -> int:
        pf = self.pixel_format.upper()
        if "16" in pf:
            return 65535
        if "14" in pf:
            return 16383
        if "12" in pf:
            return 4095
        if "10" in pf:
            return 1023
        if "8" in pf:
            return 255
        return 65535

    def _roi_view(self, frame: np.ndarray) -> np.ndarray:
        """센터 ROI 잘라서 샘플링(행/열 stride 적용)."""
        if frame is None or frame.ndim != 2:
            return frame
        h, w = frame.shape
        rh = int(h * self.roi_ratio / 2)
        rw = int(w * self.roi_ratio / 2)
        cy, cx = h // 2, w // 2
        y1, y2 = max(0, cy - rh), min(h, cy + rh)
        x1, x2 = max(0, cx - rw), min(w, cx + rw)
        return frame[y1:y2:self.rstride, x1:x2:self.cstride]

    def _measure_at(self, lb_level: int, delay: float, ctrl) -> Tuple[float, bool, np.ndarray]:
        """
        지정된 light level에서 (필요하면 TDI 워밍업 후) 한 장만 그랩해서
        ROI 상위 퍼센타일(top_percent)과 전체 평균을 계산.
        """
        if not self.lightbox.set_light_level(int(lb_level)):
            raise RuntimeError(f"Failed to set light level to {lb_level}")
        time.sleep(delay)

        # 라인/TDI 카메라라면 per-level 워밍업
        warmup = self._per_level_warmup if self.is_linescan else 0
        for _ in range(warmup):
            with suppress(Exception):
                ctrl.get_next_frame(timeout_ms=1000)

        # 실제 측정용 프레임 1장
        fa = ctrl.get_next_frame(timeout_ms=3000)
        fa = _reshape_if_rgb_unpacked_worker(fa, self.pixel_format, self.camera)

        gfa = _to_g_mono(fa, self.pixel_format, self.bayer_pattern)
        if gfa is None or gfa.ndim != 2:
            raise RuntimeError("Unexpected frame format for saturation metrics.")

        roi = self._roi_view(gfa).astype(np.float64)
        gv = roi.ravel()
        top = np.percentile(gv, 100.0 * (1.0 - self.top_percent)) if gv.size else 0.0
        max_dn = self._max_dn_from_pixfmt()
        gclip = (top >= (max_dn - self.slack))
        gmean = float(np.mean(gfa))
        return gmean, gclip, fa

    # ------------------------------------------------------------------ main
    def run(self):
        ctrl = self.camera.ctrl
        try:
            if not self.lightbox.power_control(turn_on=True):
                raise RuntimeError("Failed to turn on lightbox power.")

            lb_max = int(self.lb_max_units)
            base_delay_fast = 0.008
            base_delay_slow = 0.020
            exp_factor = 2.0
            max_exp_steps = 18
            bin_iters = 5
            linear_span_max = 64
            linear_step = 1

            step_counter = 0

            # full-scale DN (PixelFormat 기반)
            max_dn = self._max_dn_from_pixfmt()

            # ---- grab 한 번만 시작 + 초기 TDI 워밍업 ----
            with suppress(Exception):
                ctrl.wait_until_idle(timeout=0.5)
            if not ctrl.is_grabbing():
                ctrl.start_grab(buffer_count=32)
                time.sleep(0.01)

            # 조명 0에서 TDI 파이프라인 채우기
            if not self.lightbox.set_light_level(0):
                raise RuntimeError("Failed to set light level to 0 for saturation warmup.")
            time.sleep(0.05)
            for _ in range(self._initial_warmup):
                if self._stop:
                    return
                with suppress(Exception):
                    ctrl.get_next_frame(timeout_ms=1000)

            # 0 레벨 프리뷰 한 번
            fa0 = ctrl.get_next_frame(timeout_ms=3000)
            fa0 = _reshape_if_rgb_unpacked_worker(fa0, self.pixel_format, self.camera)
            if step_counter % self.preview_every == 0:
                self.preview_ready.emit(fa0)
            step_counter += 1

            # [1] 지수 탐색 (coarse) + "포화 근처 DN"에서 break
            lb = max(1, min(lb_max, int(exp_factor)))
            high = None
            exp_count = 0
            last_clip = False

            while lb <= lb_max and exp_count < max_exp_steps:
                if self._stop:
                    return

                gmean, gclip, frm = self._measure_at(lb, base_delay_fast, ctrl)
                if step_counter % self.preview_every == 0:
                    self.preview_ready.emit(frm)
                self.progress_updated.emit(0, f"Saturation search (exp) level={lb}, Gmean={gmean:.1f}")
                step_counter += 1
                exp_count += 1

                if gclip:
                    # 두 번 연속 clip이면 high 확정
                    if last_clip:
                        high = lb
                        break
                    last_clip = True
                    lb = min(lb_max, lb + 1)
                    continue

                # 아직 clip은 아님
                last_clip = False

                # 평균 DN이 full-scale의 exp_stop_ratio 이상이면
                # "포화 근처"로 보고 여기서 high 확정
                if gmean >= self.exp_stop_ratio * max_dn:
                    high = lb
                    break

                # 아직 어두운 영역 → 계속 2배수 성장
                lb = min(lb_max, max(lb + 1, int(lb * exp_factor)))

            if high is None:
                # clip/near-sat을 못 찾고 끝났으면 마지막 레벨을 high로 사용
                high = lb

            # [2] 이분 탐색
            low = max(0, high // int(exp_factor))
            for _ in range(bin_iters):
                if self._stop:
                    return
                mid = (low + high) // 2
                delay = base_delay_slow if (high - low) <= 256 else base_delay_fast
                _, gclip, frm = self._measure_at(mid, delay, ctrl)
                if step_counter % self.preview_every == 0:
                    self.preview_ready.emit(frm)
                self.progress_updated.emit(0, f"Saturation search (bin) [{low},{mid},{high}]")
                step_counter += 1
                if gclip:
                    high = mid
                else:
                    low = mid
                if high - low <= 16:
                    break

            # [3] 선형 스윕에서 last_good / first_clip 탐색
            start = max(0, high - min(linear_span_max, max(16, (high - low) * 2)))
            first_clip_units = None
            last_good_units = None
            cur = start
            while cur <= high:
                if self._stop:
                    return
                _, gclip, frm = self._measure_at(cur, base_delay_slow, ctrl)
                if step_counter % self.preview_every == 0:
                    self.preview_ready.emit(frm)
                self.progress_updated.emit(0, f"Saturation search (lin) level={cur}")
                step_counter += 1

                if gclip:
                    if first_clip_units is None:
                        first_clip_units = cur
                    # 처음 clip을 찾았으면 그 위는 전부 clip이므로 종료
                    break
                else:
                    last_good_units = cur

                cur += linear_step

            # fallback: clip이 한 번도 안 잡혀도 high를 clip로 사용
            if first_clip_units is None:
                first_clip_units = high
            if last_good_units is None:
                last_good_units = max(start, first_clip_units - 1)

            # clip level (units) → app 스케일(0~65535)로 변환
            clip_units = max(1, min(first_clip_units, lb_max))

            if self.lb_max_units >= 65535:
                clip_app = clip_units
            else:
                clip_app = int(round(clip_units * 65535.0 / self.lb_max_units))

            self.saturation_found.emit(int(clip_app))

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            try:
                if self.lightbox:
                    self.lightbox.power_control(turn_on=False)
            finally:
                with suppress(Exception):
                    if ctrl.is_grabbing():
                        ctrl.stop_grab(flush=False)
                        ctrl.wait_until_idle(timeout=0.5)
                self.finished.emit()


class MeasurementWorker(QThread, _SnapMixin):
    progress_updated = pyqtSignal(int, str)
    result_ready = pyqtSignal(dict)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self,
                 camera: CxpCamera,
                 lightbox: LightBoxController,
                 light_steps: list,
                 pixel_format: str,
                 bayer_pattern: Optional[str] = None,
                 lb_max_units: int = DEFAULT_LB_MAX_UNITS,
                 dark_frame_count: int = 16,
                 pre_last_index: Optional[int] = None):
        super().__init__()
        self.camera = camera
        self.lightbox = lightbox
        self.light_steps = [int(x) for x in light_steps]
        self.pixel_format = pixel_format or ""
        self.bayer_pattern = bayer_pattern if (bayer_pattern in VALID_BAYER) else None
        self.lb_max_units = int(lb_max_units)
        self.dark_frame_count = int(max(1, dark_frame_count))
        self.pre_last_index = pre_last_index if pre_last_index is not None else len(self.light_steps) - 1
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        ctrl = self.camera.ctrl
        try:
            if not self.lightbox.power_control(turn_on=True):
                raise RuntimeError("Failed to turn on lightbox power.")

            # ------------------------------------------------------------------
            # 1) 다크 스택: 연속 grab + 워밍업 후 steady-frame만 사용
            # ------------------------------------------------------------------
            self.progress_updated.emit(0, "Measuring dark frames…")
            if not self.lightbox.set_light_level(self._to_lb_units(0)):
                raise RuntimeError("Failed to set light level to 0 for dark frames.")
            time.sleep(0.2)

            with suppress(Exception):
                ctrl.wait_until_idle(timeout=0.5)
            if not ctrl.is_grabbing():
                ctrl.start_grab(buffer_count=32)
                time.sleep(0.01)

            WARMUP_FRAMES = 16
            for _ in range(WARMUP_FRAMES):
                if not self._is_running:
                    break
                with suppress(Exception):
                    ctrl.get_next_frame(timeout_ms=1000)

            dark_vals = []
            for i in range(self.dark_frame_count):
                if not self._is_running:
                    return

                try:
                    fa = ctrl.get_next_frame(timeout_ms=3000)
                    fb = ctrl.get_next_frame(timeout_ms=3000)
                except Exception as e:
                    raise RuntimeError(f"Dark frame grab failed: {e}") from e

                fa_rgb = _reshape_if_rgb_unpacked_worker(fa, self.pixel_format, self.camera)
                fb_rgb = _reshape_if_rgb_unpacked_worker(fb, self.pixel_format, self.camera)
                gA = _to_g_mono(fa_rgb, self.pixel_format, self.bayer_pattern)
                gB = _to_g_mono(fb_rgb, self.pixel_format, self.bayer_pattern)

                dark_vals.append(float((np.mean(gA) + np.mean(gB)) * 0.5))
                self.progress_updated.emit(
                    int(10 * (i + 1) / self.dark_frame_count),
                    f"Dark {i + 1}/{self.dark_frame_count}",
                )

            if not dark_vals:
                raise RuntimeError("No dark frames collected.")

            dv = np.sort(np.array(dark_vals))
            if len(dv) >= 4:
                k = max(1, int(len(dv) * 0.25))   # 상위 25% 버리기
                used = dv[:len(dv) - k]
            else:
                used = dv
            dark_mean = float(np.mean(used))

            self.progress_updated.emit(12, "Dark completed.")

            with suppress(Exception):
                if ctrl.is_grabbing():
                    ctrl.stop_grab(flush=False)
                    ctrl.wait_until_idle(timeout=0.5)

            # ------------------------------------------------------------------
            # 2) 본 측정: snap_pair_safe 사용 (TDI 워밍업 포함)
            # ------------------------------------------------------------------
            total_steps = len(self.light_steps)
            prev_for_plateau = None

            for i, app_light_level in enumerate(self.light_steps):
                if not self._is_running:
                    break

                # 선형 구간/검증 구간 표시
                is_verify_step = (i > self.pre_last_index)

                progress = 12 + int(88 * i / max(1, (total_steps - 1)))
                if is_verify_step:
                    msg = f"Step {i + 1}/{total_steps} (verify saturation)…"
                else:
                    msg = f"Step {i + 1}/{total_steps}..."
                self.progress_updated.emit(progress, msg)

                if not self.lightbox.set_light_level(self._to_lb_units(int(app_light_level))):
                    raise RuntimeError(f"Failed to set light level to {app_light_level}")
                time.sleep(0.02)

                fa, fb = self._snap_pair_safe()

                fa_rgb = _reshape_if_rgb_unpacked_worker(fa, self.pixel_format, self.camera)
                fb_rgb = _reshape_if_rgb_unpacked_worker(fb, self.pixel_format, self.camera)
                gA = _to_g_mono(fa_rgb, self.pixel_format, self.bayer_pattern)
                gB = _to_g_mono(fb_rgb, self.pixel_format, self.bayer_pattern)
                mean_val = float((np.mean(gA) + np.mean(gB)) * 0.5)

                std_temporal_mask = _std_temporal_green_mask(fa, fb, self.bayer_pattern)
                if std_temporal_mask is not None:
                    std_temporal = float(std_temporal_mask)
                else:
                    std_temporal = float(
                        np.std((gA.astype(np.float64) - gB.astype(np.float64)) / np.sqrt(2.0))
                    )

                var_single = _spatial_var_green_mask(fb, self.bayer_pattern, remove_dc=True, ddof=0)
                if var_single is not None:
                    total_std = float(np.sqrt(var_single))
                else:
                    gb0 = gB.astype(np.float64) - float(np.mean(gB))
                    total_std = float(np.sqrt(np.var(gb0)))

                if int(app_light_level) == 0:
                    gray_sub = mean_val
                    gray_display = mean_val
                else:
                    gray_sub = mean_val - dark_mean
                    gray_display = gray_sub

                # plateau 판정 (기존 로직)
                plateau = False
                if prev_for_plateau is not None:
                    inc_ratio = (mean_val - prev_for_plateau) / max(prev_for_plateau, 1e-12)
                    inc_abs = (mean_val - prev_for_plateau)
                    plateau = (inc_ratio < 1e-3) or (inc_abs < 0.5)
                prev_for_plateau = mean_val

                self.result_ready.emit({
                    "light_level": float(app_light_level),
                    "gray_value_raw": float(mean_val),
                    "gray_value_subtracted": float(gray_sub),
                    "gray_value_display": float(gray_display),
                    "temporal_std": float(std_temporal),
                    "total_std": float(total_std),
                    "last_frame": fb_rgb,
                    "is_plateau": bool(plateau),
                    "is_verify_step": bool(is_verify_step),  # ★ 추가: 선형 vs 포화 확인용
                })

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            try:
                if self.lightbox:
                    self.lightbox.power_control(turn_on=False)
            finally:
                with suppress(Exception):
                    if ctrl.is_grabbing():
                        ctrl.stop_grab(flush=False)
                self.finished.emit()

