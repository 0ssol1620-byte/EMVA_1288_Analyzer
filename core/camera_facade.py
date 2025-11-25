# core/camera_facade.py

from PyQt5.QtCore import QObject, pyqtSignal
from .camera_controller import CameraController
import numpy as np
import time
from typing import Optional, Tuple
from contextlib import suppress


class CxpCamera(QObject):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, ctrl: Optional[CameraController] = None):
        super().__init__()
        self.ctrl = ctrl or CameraController()
        # CameraController에서 reshape된 프레임을 바로 받는다.
        self.ctrl.reshaped_frame_ready.connect(self.new_frame)

    # ---------- high-level API ----------
    def connect(self) -> bool:
        cams = self.ctrl.discover_cameras()
        if not cams:
            return False
        return self.ctrl.connect_camera_by_info(cams[0]["camera_info"])

    def set(self, name: str, value):
        self.ctrl.set_param(name, value)

    def get(self, name: str):
        return self.ctrl.get_param(name)

    def start_preview(self):
        self.ctrl.start_live_view()

    def stop_preview(self):
        self.ctrl.stop_live_view()

    def snap_pair(self, delay_ms: int = 0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        프리뷰/라이브뷰용 간단 snap:
        - grab 이 안 켜져 있으면 buffer_count=8로 시작
        - 프레임 2장만 연속으로 읽고,
        - 우리가 시작한 grab만 stop_grab(flush=False)로 정리.
        ※ TDI 워밍업은 Dark/Linearity 워커 내부에서만 처리.
        """
        ctrl = self.ctrl
        if ctrl is None:
            return None, None

        pre_started = False
        if not ctrl.is_grabbing():
            ctrl.start_grab(buffer_count=8)
            pre_started = True

        frame1 = None
        frame2 = None
        try:
            frame1 = ctrl.get_next_frame(timeout_ms=3000)
            if delay_ms:
                time.sleep(delay_ms / 1000.0)
            frame2 = ctrl.get_next_frame(timeout_ms=3000)
        finally:
            if pre_started:
                with suppress(Exception):
                    ctrl.stop_grab(flush=False)   # GRACEFUL 정지

        return frame1, frame2

    # convenience
    def width(self) -> int:
        return int(self.get("Width"))

    def height(self) -> int:
        return int(self.get("Height"))
