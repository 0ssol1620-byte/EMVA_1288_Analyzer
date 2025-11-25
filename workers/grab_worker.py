#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GrabWorker – resilient multi‑camera frame drainer
================================================
* Python 3.8+ & PyQt5 compatible (Qt6: only signal type hints need tweaks)
* Designed to **keep DMA/FIFO queues empty** so the driver never overruns
* Emits two signals:
    1. ``CameraController.frame_ready`` – raw ``np.ndarray`` (backwards‑compat)
    2. ``GrabWorker.frame_ready``      – ready‑to‑paint ``QImage``
* Adaptive timeout & batch draining (``drain_limit``) minimise kernel hops
* Optional FPS / loss statistics when SDK supplies ``frame.id`` metadata

Usage
-----
>>> worker = GrabWorker(ctrl, target_width=640, max_fps=30)
>>> worker.frame_ready.connect(on_qimage)
>>> worker.start()
...
>>> worker.stop()
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
from PyQt5 import QtCore  # type: ignore
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage
from contextlib import suppress
if TYPE_CHECKING:
    from core.camera_controller import CameraController  # pragma: no cover
else:
    CameraController = Any

logger = logging.getLogger(__name__)

__all__ = ["GrabWorker"]


class GrabWorker(QThread):
    """Background thread continuously draining frames from one or more
    :class:`~src.core.camera_controller.CameraController` instances.

    Parameters
    ----------
    controllers : Union[CameraController, List[CameraController]]
        One or many camera controllers.
    target_width : int | None, default None
        If given, frames are down‑scaled to this width (keeping aspect).
    max_fps : int, default 60
        Upper FPS bound *per controller*. ``<=0`` disables throttling.
    drain_limit : int, default 8
        How many frames to dequeue per controller **per tick**. 1‑8 is typical.
    stats_interval : float, default 2.0
        Seconds between debug log lines with FPS/loss statistics.
    parent : QObject | None
        Optional Qt parent.
    """

    # cam_id, QImage (may be a dummy if GUI side ignores it)
    frame_ready = pyqtSignal(str, QImage)
    finished = pyqtSignal()

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        controllers: Union[List["CameraController"], "CameraController"],
        *,
        target_width: Optional[int] = None,
        max_fps: int = 60,
        drain_limit: int = 8,
        stats_interval: float = 2.0,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._controllers: List[CameraController] = (
            controllers if isinstance(controllers, list) else [controllers]
        )
        self._target_w: int = max(64, target_width) if target_width else 0
        self._frame_interval: float = 0.0 if max_fps <= 0 else 1.0 / max_fps
        # adaptive timeout: ≥50 ms or 3×frame interval
        base_ms = int(self._frame_interval * 3000) if self._frame_interval else 100
        self._timeout_ms: int = max(50, base_ms)  # ← was 20 ms
        self._drain_limit: int = max(1, drain_limit)
        self._stats_interval: float = max(0.5, stats_interval)
        self._stats: dict[str, dict[str, float | int | None]] = {}
        self._running = threading.Event()
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- helpers
    @property
    def controllers(self) -> List["CameraController"]:
        """Mutable list of managed controllers (add/remove on the fly)."""
        return self._controllers

    @pyqtSlot(int)
    def update_target_width(self, w: int) -> None:
        """Live‑update thumbnail width (0 disables scaling)."""
        self._target_w = max(64, w) if w > 0 else 0

    # ───────────────── Thread-safe controller management ─────────────────
    def add_controller(self, controller: "CameraController") -> None:
        with self._lock:
            if controller not in self._controllers:
                self._controllers.append(controller)
                logger.debug("GrabWorker: Added controller %s", getattr(controller, 'cam_id', 'N/A'))

    def remove_controller(self, controller: "CameraController") -> None:
        with self._lock:
            try:
                self._controllers.remove(controller)
                logger.debug("GrabWorker: Removed controller %s", getattr(controller, 'cam_id', 'N/A'))
            except ValueError:
                pass # Already removed

    def stop(self, timeout_ms: int = 2000) -> None:
        """Request the thread to quit and wait up to *timeout_ms*."""
        self._running.clear()
        if self.isRunning():
            if not self.wait(timeout_ms):
                logger.warning("GrabWorker thread did not stop gracefully. Terminating.")
                self.terminate()
                self.wait() # Wait for termination to complete

    def run(self) -> None:
        logger.debug("GrabWorker started (controllers=%d)", len(self.controllers))
        self._running.set()

        while self._running.is_set():
            with self._lock:
                live_controllers = list(self._controllers)

            if not live_controllers:
                time.sleep(0.1)
                continue

            for ctrl in live_controllers:
                if not (ctrl.is_connected() and ctrl.is_grabbing()):
                    continue

                for _ in range(self._drain_limit):
                    if not self._running.is_set(): break
                    try:
                        # *** KEY CHANGE HERE ***
                        # Use count_timeout_error=False to prevent normal polling
                        # timeouts from being logged as statistical errors.
                        frame = ctrl.get_next_frame(
                            timeout_ms=self._timeout_ms,
                            count_timeout_error=False,
                        )
                    except Exception:
                        break  # Move to next controller

                    if frame is None:
                        break  # Clean timeout, no more frames for now

                    self._update_stats(ctrl.cam_id, frame)

                    with suppress(Exception):
                        ctrl.frame_ready.emit(frame)

                    view = self._resize_if_needed(frame)
                    if (qimg := self._ndarray_to_qimage(view)) is not None:
                        with suppress(Exception):
                            self.frame_ready.emit(ctrl.cam_id, qimg)

                if not self._running.is_set(): break

            time.sleep(0.001)

        self.finished.emit()
        logger.info("GrabWorker thread finished.")

    # ----------------------------------------------------- internal helpers
    def _update_stats(self, cam_id: str, frame: np.ndarray) -> None:
        st = self._stats.setdefault(
            cam_id, dict(recv=0, lost=0, last_id=None, last_ts=time.perf_counter())
        )
        st["recv"] += 1
        fid = getattr(frame, "id", None)
        if fid is not None and st["last_id"] is not None:
            gap = fid - st["last_id"] - 1
            if gap > 0:
                st["lost"] += gap
        st["last_id"] = fid

        now = time.perf_counter()
        if now - st["last_ts"] >= self._stats_interval:
            span = now - st["last_ts"]
            fps = st["recv"] / span if span else 0.0
            total = st["recv"] + st["lost"]
            loss_pct = (st["lost"] / total * 100) if total else 0.0
            logger.debug("[%s] Stats: fps=%.2f  loss=%.2f%%", cam_id, fps, loss_pct)
            st.update(recv=0, lost=0, last_ts=now)

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        if (
            self._target_w
            and frame.ndim >= 2
            and frame.shape[1] > self._target_w
        ):
            try:
                import cv2  # lazy import keeps import time low if not used

                scale = self._target_w / frame.shape[1]
                new_h = int(frame.shape[0] * scale)
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                return cv2.resize(frame, (self._target_w, new_h), interpolation=interp)
            except Exception:
                pass
        return frame

    # ------------------------------------------------ ndarray → QImage
    @staticmethod
    def _ndarray_to_qimage(arr: np.ndarray) -> Optional[QImage]:
        try:
            # Grayscale 8/16‑bit → 8‑bit
            if arr.ndim == 2:
                if arr.dtype == np.uint16:
                    arr = (arr >> 8).astype(np.uint8)
                if arr.dtype != np.uint8:
                    return None
                h, w = arr.shape
                return QImage(arr.data, w, h, arr.strides[0], QImage.Format_Grayscale8).copy()

            # BGR/BGRA → RGB/RGBA
            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                if arr.dtype != np.uint8:
                    arr = np.clip(arr >> 8, 0, 255).astype(np.uint8)
                if arr.shape[2] == 3:
                    arr = arr[..., ::-1].copy()
                    fmt = QImage.Format_RGB888
                else:
                    arr = arr[..., [2, 1, 0, 3]].copy()
                    fmt = QImage.Format_RGBA8888
                h, w, ch = arr.shape
                return QImage(arr.data, w, h, w * ch, fmt).copy()
        except Exception as exc:
            logger.debug("ndarray→QImage failed: %s", exc, exc_info=True)
        return None
