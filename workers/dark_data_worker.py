# workers/dark_data_worker.py
# -*- coding: utf-8 -*-

import time
import numpy as np
import contextlib
from typing import List, Optional, Tuple
from PyQt5.QtCore import QThread, pyqtSignal

from core.camera_facade import CxpCamera
from core.camera_exceptions import FrameAcquisitionError


class DarkDataWorker(QThread):
    """
    EMVA 1288 Dark / Read-Noise 측정 (라인/TDI 대응 완전판)

    - EMVA pairwise 정의를 O(N) 폐형식으로 정확 계산 (전쌍과 동치, 속도 ↑)
      σ_read^2 = 0.5 * E_pairs[(I_i - I_j)^2] = (n * Σx^2 - (Σx)^2) / (n*(n-1))

    - Total Noise(DN): 프레임별 DC 제거 공간 표준편차의 평균
    - threshold_dn 초과 픽셀 NaN 마스킹
    - RGB→G, Bayer→G-compact(Gr/Gb 평균), Mono→그대로

    - Area 카메라:
        ExposureTime(µs) 를 직접 제어

    - Linescan / TDI:
        ExposureTime 대신 AcquisitionLineRate 제어
        • 입력 exposure_steps[µs] 는 "원하는 유효 적분시간 T_int" 로 해석
        • T_int ≈ effective_stages / LineRate
          (effective_stages = TDIStages 있으면 그 값, 없으면 Height)

        • LineRate 는 GenICam 노드:
            AcquisitionLineRateMin / AcquisitionLineRateMax
          범위 안으로 클램프
        • 테스트 중 카메라: 최소 30 kHz → 하드 최소값도 30kHz 보장

    - 견고성: GenTL/-1/Busy/timeout 감지 시 단계적 복구
      (soft → hard → TL reset → 재시도)

    - 프레임 크기 불일치: 공통 최소 크기로 정규화

    - 결과 필드 예:
        {
          "method": "pairwise",
          "exposure_time_s", "exposure_time_us",   # effective T_int
          "mean_gray_value",
          "read_noise_dn", "read_noise_dn_median",
          "read_noise_e",  "read_noise_e_median",
          "total_noise_dn"
        }
    """

    progress_updated = pyqtSignal(int, str)
    result_ready = pyqtSignal(dict)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        camera: CxpCamera,
        exposure_steps: List[float],              # microseconds (effective T_int)
        pixel_format: str,
        frames_to_average: int = 16,             # ≥2
        system_gain_e_per_dn: Optional[float] = None,
        compute_per_pixel_pairwise: bool = False,  # threshold 사용 시 맵 생성은 비활성
        threshold_dn: Optional[float] = None,
    ):
        super().__init__()
        self.camera = camera
        self.exposure_steps = [float(x) for x in exposure_steps]
        self.pixel_format = pixel_format or ""
        self.frames_to_average = max(2, int(frames_to_average))
        self.system_gain = float(system_gain_e_per_dn) if system_gain_e_per_dn else None
        self.compute_per_pixel_pairwise = bool(compute_per_pixel_pairwise)
        self.threshold_dn = (float(threshold_dn) if threshold_dn is not None else None)

        self.is_rgb = "RGB" in self.pixel_format
        self.is_bayer = "Bayer" in self.pixel_format
        self.is_linescan = False
        with contextlib.suppress(Exception):
            self.is_linescan = (str(self.camera.get("DeviceScanType") or "") == "Linescan")

        # ⬇️ TDI 스테이지 & 워밍업 프레임 계산 추가 -------------------
        self._tdi_stages: int = 0
        if self.is_linescan:
            with contextlib.suppress(Exception):
                val = self.camera.get("TDIStages")
                if isinstance(val, str):
                    digits = "".join(ch for ch in val.split()[-1] if ch.isdigit())
                    if digits:
                        self._tdi_stages = int(digits)
                elif isinstance(val, (int, float)):
                    code = int(val)
                    stage_map = {
                        1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 96,
                        7: 128, 8: 160, 9: 192, 10: 224,
                        11: 240, 12: 248, 13: 252, 14: 256,
                    }
                    self._tdi_stages = stage_map.get(code, 0)

        # TDI면 2×스테이지, 아니면 최소 2프레임만 버리도록
        if self.is_linescan and self._tdi_stages > 1:
            self._warmup_frames = max(2 * self._tdi_stages, 16)
        else:
            self._warmup_frames = 2   # non-TDI 기본값

        self._is_running = True
    # ------------------------------ control ------------------------------ #
    def stop(self):
        self._is_running = False

    # ------------------------------- main -------------------------------- #
    def run(self):
        orig_exp = None
        orig_lr = None
        try:
            # 원래 Exposure / LineRate 백업
            with contextlib.suppress(Exception):
                if self.is_linescan:
                    orig_lr = float(self.camera.get("AcquisitionLineRate"))
                else:
                    orig_exp = float(self.camera.get("ExposureTime"))

            h_nom, w_nom = self._get_hw_nominal()
            total_steps = max(1, len(self.exposure_steps))
            self.progress_updated.emit(0, "Starting Dark & Read-Noise measurement...")

            for i, exp_us in enumerate(self.exposure_steps):
                if not self._is_running:
                    break

                # clean idle
                self._graceful_idle()

                # 입력 exp_us(µs)는 "원하는 유효 적분시간 T_int" 로 해석
                exp_s = max(0.0, float(exp_us) / 1_000_000.0)

                if self.is_linescan:
                    # 라인/TDI 카메라: T_int ≈ effective_stages / LineRate
                    lr = self._choose_line_rate(exp_s, h_nom)
                    self._set_line_rate(lr)
                else:
                    # Area 카메라: ExposureTime 그대로 사용
                    self._set_exposure_us(float(exp_us))

                time.sleep(0.10)

                timeout_s = self._estimate_timeout(exp_s, h_nom)
                self.progress_updated.emit(
                    int(100 * i / total_steps),
                    f"[Dark] Step {i+1}/{total_steps}  T_int≈{exp_s:.6f}s  timeout={timeout_s:.2f}s"
                )

                # start grab + warm-up discard
                # start grab + warm-up discard
                self._start_grab_safe(32)
                warmup = self._warmup_frames if self.is_linescan else 2
                self._discard_frames(warmup, timeout_s)

                # capture planes (mono/G), NaN-mask by threshold
                planes, frame_means = self._collect_dark_planes(
                    h_nom, w_nom, timeout_s, exp_s, i, total_steps, self.threshold_dn
                )

                # stop and idle
                self._graceful_idle()

                # stats
                mean_dn = float(np.nanmean(frame_means)) if frame_means else None

                total_std_list = []
                for p in planes:
                    if p is None or np.isnan(p).all():
                        total_std_list.append(np.nan)
                        continue
                    pm = p - np.nanmean(p)
                    total_std_list.append(float(np.nanstd(pm)))
                total_noise_dn = float(np.nanmean(total_std_list)) if total_std_list else None

                rn_dn, rn_dn_median, rn_e, rn_e_median, rn_map = self._read_noise_pairwise_O1(planes)

                if mean_dn is not None and rn_dn is not None:
                    result = {
                        "method": "pairwise",
                        "exposure_time_us": float(exp_us),   # 입력 T_int(µs)
                        "exposure_time_s": float(exp_s),     # 입력 T_int(s)
                        "mean_gray_value": float(mean_dn),
                        "read_noise_dn": float(rn_dn),
                        "read_noise_dn_median": float(rn_dn_median) if rn_dn_median is not None else None,
                        "total_noise_dn": float(total_noise_dn) if total_noise_dn is not None else None,
                    }
                    if self.system_gain is not None:
                        result["read_noise_e"] = float(rn_e) if rn_e is not None else None
                        result["read_noise_e_median"] = float(rn_e_median) if rn_e_median is not None else None
                    if rn_map is not None:
                        result["read_noise_map"] = rn_map
                    self.result_ready.emit(result)

        except Exception as e:
            self.error_occurred.emit(f"An error occurred in Dark Data worker: {e}")
        finally:
            # 원래 Exposure / LineRate 복원
            with contextlib.suppress(Exception):
                if self.is_linescan and orig_lr is not None:
                    self.camera.set("AcquisitionLineRate", float(orig_lr))
                if (not self.is_linescan) and orig_exp is not None:
                    self.camera.set("ExposureTime", float(orig_exp))
            self.finished.emit()

    # -------------------------- camera utilities -------------------------- #
    def _get_hw_nominal(self) -> Tuple[int, int]:
        h = 0
        w = 0
        with contextlib.suppress(Exception):
            h = int(self.camera.get("Height"))
        with contextlib.suppress(Exception):
            w = int(self.camera.get("Width"))
        return max(1, h), max(1, w)

    def _set_exposure_us(self, exp_us: float):
        self.camera.set("ExposureTime", float(max(1.0, exp_us)))

    # ---- LineRate 범위: GenICam Feature 기반 + 하드 최소 30k 보장 ---- #
    def _get_line_rate_limits(self) -> Tuple[float, float]:
        """
        AcquisitionLineRate 의 Min/Max 를 안정적으로 읽어서 (min,max) Hz 반환.
        우선순위:
          1) CameraController.get_parameter_metadata("AcquisitionLineRate") 의 min/max
          2) AcquisitionLineRateMinReg / MaxReg (XML 레지스터)
          3) 현재 AcquisitionLineRate 기준으로 +/-50% 추정
          4) 최종 하드 범위: min ≥ 30 kHz
        """
        HARD_MIN_LR = 30_000.0  # 이 카메라 최소 스펙
        lr_min = None
        lr_max = None

        # 1) 메타데이터 기반 (가장 신뢰도 높음)
        with contextlib.suppress(Exception):
            meta = self.camera.ctrl.get_parameter_metadata("AcquisitionLineRate")
            if meta:
                mmin = meta.get("min")
                mmax = meta.get("max")
                if isinstance(mmin, (int, float)) and 1.0 <= mmin <= 1e9:
                    lr_min = float(mmin)
                if isinstance(mmax, (int, float)) and 1.0 <= mmax <= 1e9:
                    lr_max = float(mmax)

        # 2) 필요 시 Reg 노드 보조 사용
        if lr_min is None:
            with contextlib.suppress(Exception):
                vmin_reg = self.camera.get("AcquisitionLineRateMinReg")
                if vmin_reg is not None and 1.0 <= float(vmin_reg) <= 1e9:
                    lr_min = float(vmin_reg)
        if lr_max is None:
            with contextlib.suppress(Exception):
                vmax_reg = self.camera.get("AcquisitionLineRateMaxReg")
                if vmax_reg is not None and 1.0 <= float(vmax_reg) <= 1e9:
                    lr_max = float(vmax_reg)

        # 3) 둘 다 못 읽었으면 현재 값 기준 추정
        cur_lr = None
        with contextlib.suppress(Exception):
            cur_lr = float(self.camera.get("AcquisitionLineRate"))
        if not cur_lr or cur_lr <= 0:
            cur_lr = 100_000.0  # 완전 실패 시 디폴트

        if lr_min is None:
            lr_min = max(HARD_MIN_LR, cur_lr * 0.5)
        if lr_max is None:
            lr_max = cur_lr * 1.5

        # 4) sanity & 하드 범위 적용
        if lr_min < HARD_MIN_LR:
            lr_min = HARD_MIN_LR
        if lr_max <= lr_min:
            lr_max = lr_min

        print(f"[DarkWorker] LRmin={lr_min:.3f} Hz, LRmax={lr_max:.3f} Hz")
        return lr_min, lr_max

    def _choose_line_rate(self, exp_s: float, height_lines: int) -> float:
        """
        원하는 '유효 적분시간(exp_s)' 에 맞춰 라인레이트를 선택.

        - Linescan/TDI 의 경우:
            T_int ≈ effective_stages / LineRate
            → LineRate ≈ effective_stages / T_int

          effective_stages:
            - TDIStages 가 있으면 그 값(self._tdi_stages)
            - 없으면 Height(라인 수)를 사용
        """
        lr_min, lr_max = self._get_line_rate_limits()

        if exp_s <= 0.0:
            return float((lr_min + lr_max) * 0.5)

        est = 1.0 / max(exp_s, 1e-9)

        lr = float(min(lr_max, max(lr_min, est)))
        return lr


    def _set_line_rate(self, lr: float):
        self.camera.set("AcquisitionLineRate", float(lr))

    def _estimate_timeout(self, exp_s: float, height_lines: int) -> float:
        if not self.is_linescan:
            return max(3.0, exp_s * 3.0 + 0.5)
        line_rate = None
        with contextlib.suppress(Exception):
            line_rate = float(self.camera.get("AcquisitionLineRate"))
        if not line_rate or line_rate <= 0:
            return 5.0
        frame_time = float(height_lines) / float(line_rate)
        return max(3.0, frame_time * 2.0 + 0.5)

    def _graceful_idle(self):
        ctrl = self.camera.ctrl
        with contextlib.suppress(Exception):
            if ctrl.is_grabbing():
                ctrl.stop_grab(flush=False)
        with contextlib.suppress(Exception):
            ctrl.flush_buffers(block=True, timeout=0.8)
        with contextlib.suppress(Exception):
            ctrl.wait_until_idle(timeout=0.8)
        # TL 레벨 버벅임 예방: 아주 짧게 쉼
        time.sleep(0.02)

    # ------------------------ start/stop & frame fetch -------------------- #
    def _start_grab_safe(self, buffer_count: int = 32):
        """
        start_grab() 단계에서 발생하는 GenTL/-1/Busy 를 감지해서
        ① soft retry → ② stream hard reset → ③ TL reset 후 재시도.
        그래도 안 되면 예외를 그대로 올려서 상위에서 에러 팝업.
        """
        ctrl = self.camera.ctrl
        with contextlib.suppress(Exception):
            ctrl.wait_until_idle(timeout=0.5)

        if ctrl.is_grabbing():
            # 이미 다른 곳에서 그랩 중이면 그냥 사용
            return

        soft_retries = 2
        last_err = None

        for attempt in range(soft_retries + 1):
            try:
                ctrl.start_grab(buffer_count=buffer_count, min_buffers=32)
                time.sleep(0.01)
                return
            except Exception as e:
                last_err = e
                msg = str(e)
                transient = (
                    "GenTL" in msg
                    or "-1" in msg
                    or "GC_ERR_BUSY" in msg
                    or "busy" in msg.lower()
                    or "timeout" in msg.lower()
                )

                # 1) soft retry
                if attempt < soft_retries and transient:
                    with contextlib.suppress(Exception):
                        ctrl.stop_grab(flush=True)
                    with contextlib.suppress(Exception):
                        ctrl.flush_buffers(block=True, timeout=1.0)
                    with contextlib.suppress(Exception):
                        ctrl.wait_until_idle(timeout=1.0)
                    time.sleep(0.05)
                    continue

                # 2) 마지막 시도에서도 실패 → TL 레벨 리셋 한 번 더
                if transient:
                    with contextlib.suppress(Exception):
                        if hasattr(self.camera, "stop_stream"):
                            self.camera.stop_stream()
                    with contextlib.suppress(Exception):
                        if hasattr(self.camera, "flush_stream"):
                            self.camera.flush_stream()
                    with contextlib.suppress(Exception):
                        if hasattr(self.camera, "start_stream"):
                            self.camera.start_stream()
                    time.sleep(0.05)
                    try:
                        ctrl.start_grab(buffer_count=buffer_count, min_buffers=32)
                        time.sleep(0.01)
                        return
                    except Exception as e2:
                        last_err = e2

                # 여기까지 오면 회복 불가로 보고 예외 올림
                raise last_err

    def _discard_frames(self, n: int, timeout_s: float):
        for _ in range(max(0, n)):
            if not self._is_running:
                return
            self._get_next_frame_with_recovery(timeout_s, warmup=True)

    def _get_next_frame_with_recovery(self, timeout_s: float, warmup: bool = False):
        """
        robust frame fetch with TL recovery (GenTL -1 / Busy / timeout 등)
        """
        ctrl = self.camera.ctrl
        timeout_ms = int(timeout_s * 1000)
        soft_retries = 2
        last_err = None

        for attempt in range(soft_retries + 1):
            try:
                f = ctrl.get_next_frame(timeout_ms=timeout_ms)
                if f is None:
                    raise FrameAcquisitionError("Timeout (None frame).")
                return f
            except Exception as e:
                last_err = e
                msg = str(e)
                transient = (
                    "timeout" in msg.lower()
                    or "busy" in msg.lower()
                    or "GC_ERR_BUSY" in msg
                    or "GenTL" in msg
                    or "-1" in msg
                )
                if attempt < soft_retries and transient:
                    time.sleep(0.02)
                    continue
                # 하드 복구 단계 1: grab 재시작
                self._hard_recover_stream(timeout_ms)
                try:
                    f = ctrl.get_next_frame(timeout_ms=timeout_ms)
                    if f is None:
                        raise FrameAcquisitionError("Timeout after hard recover.")
                    return f
                except Exception as e2:
                    last_err = e2
                    # 하드 복구 단계 2: TL 레벨 리셋
                    self._reset_transport_layer()
                    f = ctrl.get_next_frame(timeout_ms=timeout_ms)
                    if f is None:
                        raise FrameAcquisitionError("Timeout after TL reset.")
                    return f

        raise FrameAcquisitionError(f"Frame fetch failed: {last_err}")

    def _hard_recover_stream(self, timeout_ms: int):
        ctrl = self.camera.ctrl
        with contextlib.suppress(Exception):
            ctrl.stop_grab(flush=True)
        time.sleep(0.05)
        with contextlib.suppress(Exception):
            ctrl.flush_buffers(block=True, timeout=timeout_ms / 1000.0)
        with contextlib.suppress(Exception):
            ctrl.wait_until_idle(timeout=timeout_ms / 1000.0)
        with contextlib.suppress(Exception):
            ctrl.start_grab(buffer_count=32)
        time.sleep(0.02)

    def _reset_transport_layer(self):
        """
        가능한 경우: 카메라/스트림 레벨 재시작.
        - camera.flush_stream / stop_stream / start_stream 지원 시 사용
        - 미지원 시 grab 재시작만으로 한정
        """
        with contextlib.suppress(Exception):
            if hasattr(self.camera, "stop_stream"):
                self.camera.stop_stream()
        with contextlib.suppress(Exception):
            if hasattr(self.camera, "flush_stream"):
                self.camera.flush_stream()
        with contextlib.suppress(Exception):
            if hasattr(self.camera, "start_stream"):
                self.camera.start_stream()
        self._hard_recover_stream(timeout_ms=1000)

    # ----------------------------- acquisition ---------------------------- #
    def _collect_dark_planes(
        self,
        h_nom: int,
        w_nom: int,
        timeout_s: float,
        exp_s: float,
        step_idx: int,
        total_steps: int,
        threshold_dn: Optional[float],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        단일 채널 평면(모노/G) 수집, 프레임 크기 동형화, 임계값 마스킹, NaN-안전 평균.
        """
        planes: List[np.ndarray] = []
        frame_means: List[float] = []
        target_h: Optional[int] = None
        target_w: Optional[int] = None

        for j in range(self.frames_to_average):
            if not self._is_running:
                break

            overall = (step_idx + (j + 1) / self.frames_to_average) / total_steps
            self.progress_updated.emit(
                int(100 * overall),
                f"T_int≈{exp_s:.6f}s - Capturing frame {j+1}/{self.frames_to_average}"
            )

            frame = self._get_next_frame_with_recovery(timeout_s)
            frame = self._reshape_frame(frame, h_nom, w_nom)
            plane = self._extract_mono_plane(frame).astype(np.float64, copy=False)

            if target_h is None or target_w is None:
                target_h, target_w = plane.shape
                target_h = max(2, int(target_h))
                target_w = max(2, int(target_w))

            hh = min(target_h, plane.shape[0])
            ww = min(target_w, plane.shape[1])
            if hh < 2 or ww < 2:
                plane_c = np.full((target_h, target_w), np.nan, dtype=np.float64)
            else:
                plane_c = plane[:hh, :ww]
                if (hh < target_h) or (ww < target_w):
                    target_h, target_w = hh, ww
                    for k in range(len(planes)):
                        planes[k] = planes[k][:target_h, :target_w]

            if threshold_dn is not None and not np.isnan(plane_c).all():
                mask = (plane_c <= threshold_dn)
                if not np.any(mask):
                    plane_c = np.full_like(plane_c, np.nan, dtype=np.float64)
                    frame_means.append(np.nan)
                else:
                    pc = plane_c.copy()
                    pc[~mask] = np.nan
                    plane_c = pc
                    frame_means.append(float(np.nanmean(plane_c)))
            else:
                frame_means.append(float(np.mean(plane_c)) if not np.isnan(plane_c).all() else np.nan)

            planes.append(plane_c)

        planes = self._crop_planes_to_common_min(planes)
        return planes, frame_means

    # --------------------- pairwise (O(N)) exact implementation ------------ #
    def _read_noise_pairwise_O1(
        self, planes: List[np.ndarray]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[np.ndarray]]:
        n_frames = len(planes)
        if n_frames < 2:
            return None, None, None, None, None

        planes = self._crop_planes_to_common_min(planes)
        if not planes:
            return None, None, None, None, None

        H, W = planes[0].shape
        stack = np.empty((n_frames, H, W), dtype=np.float64)
        for i, p in enumerate(planes):
            stack[i] = p if (p is not None) else np.nan

        valid = ~np.isnan(stack)
        n_eff = valid.sum(axis=0).astype(np.float64)

        with np.errstate(invalid="ignore"):
            S1 = np.nansum(stack, axis=0)
            S2 = np.nansum(stack * stack, axis=0)
            num = n_eff * S2 - S1 * S1
            den = n_eff * (n_eff - 1.0)
            pair_mean = np.where(n_eff >= 2.0, 2.0 * num / den, np.nan)
            rn2 = 0.5 * pair_mean
            rn = np.sqrt(np.maximum(rn2, 0.0))

        rn_dn = float(np.nanmean(rn)) if np.isfinite(rn).any() else None
        rn_dn_median = float(np.nanmedian(rn)) if np.isfinite(rn).any() else None

        rn_map = None
        if self.compute_per_pixel_pairwise and self.threshold_dn is None:
            rn_map = rn

        rn_e = (self.system_gain * rn_dn) if (self.system_gain is not None and rn_dn is not None) else None
        rn_e_median = (self.system_gain * rn_dn_median) if (
            self.system_gain is not None and rn_dn_median is not None) else None

        return rn_dn, rn_dn_median, rn_e, rn_e_median, rn_map

    # ------------------------ reshape & mono extract ---------------------- #
    def _reshape_frame(self, frame: np.ndarray, height: int, width: int) -> np.ndarray:
        if frame is None:
            raise FrameAcquisitionError("Empty frame after acquisition.")
        if self.is_rgb:
            if frame.ndim != 3 or frame.shape[2] != 3:
                if frame.dtype != np.uint16:
                    frame = frame.view(np.uint16)
                frame = frame.reshape((height, width, 3))
        else:
            if frame.ndim == 1:
                frame = frame.reshape((height, width))
        return frame

    def _extract_mono_plane(self, frame: np.ndarray) -> np.ndarray:
        if self.is_rgb:
            return frame[:, :, 1].astype(np.float64, copy=False)
        if self.is_bayer:
            gr = frame[0::2, 1::2]
            gb = frame[1::2, 0::2]
            return (gr.astype(np.float64) + gb.astype(np.float64)) * 0.5
        return frame.astype(np.float64, copy=False)

    # ------------------------------ size utils ---------------------------- #
    @staticmethod
    def _crop_planes_to_common_min(planes: List[np.ndarray]) -> List[np.ndarray]:
        shapes = [(p.shape[0], p.shape[1]) for p in planes
                  if isinstance(p, np.ndarray) and p.ndim == 2 and p.size > 0]
        if not shapes:
            return planes
        h_min = min(h for h, _ in shapes)
        w_min = min(w for _, w in shapes)
        if h_min < 2 or w_min < 2:
            return planes
        out = []
        for p in planes:
            if isinstance(p, np.ndarray) and p.ndim == 2 and p.size > 0:
                out.append(p[:h_min, :w_min])
            else:
                out.append(p)
        return out
