#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ctypes_lightbox_final_v6.py

ZIG_COMMAND enum을 통해 발견된 정확한 명령 코드를 적용한 최종 완성 버전.
- Power ON (0x0B) -> Set Brightness (0x10) 시퀀스로 동작
"""
import sys
import time
import ctypes
import threading
import argparse
from ctypes import c_void_p, c_ulong, c_char_p, c_ubyte, POINTER, byref

# ----------------------------
# ZIG_COMMAND 상수 정의
# ----------------------------
# 프로토콜 프레임
_STX1, _STX2, _END = 0x01, 0xA5, 0xFE

# 정확한 명령 코드 (ZIG_COMMAND enum 기반)
_CMD_POWER_ON = 0x0B  # _SPO (Power_ON)
_CMD_SET_BRIGHTNESS = 0x10  # _SLB (Lighting Bright)

# FTDI D2XX 상수
FT_OK = 0;
FT_LIST_NUMBER_ONLY = 0x80000000;
FT_LIST_BY_INDEX = 0x40000000
FT_OPEN_BY_DESCRIPTION = 2;
FT_PURGE_RX = 1;
FT_PURGE_TX = 2


# ----------------------------
# 예외 클래스 정의
# ----------------------------
class LightBoxError(Exception): pass


class FtdiDllError(LightBoxError): pass


class DeviceNotFoundError(LightBoxError): pass


# ------------------------------------------------------------------
# 최종 완성된 Controller 클래스
# ------------------------------------------------------------------
class LightBoxController:
    # ... (init, _define_ftdi_functions, _open_device_by_description, _read_data_loop은 이전 v5와 동일)
    def __init__(self, *, description_keyword: str = "Camera-Tester", debug: bool = False):
        self.debug = debug;
        self.ft_handle = c_void_p();
        self._is_open = False
        self._stop_thread_event = threading.Event();
        self.rx_buffer = bytearray();
        self.rx_lock = threading.Lock()
        self.reader_thread = None
        if self.debug: print("[DEBUG] Initializing LightBoxController...")
        try:
            self.ft_dll = ctypes.windll.Ftd2xx
        except OSError:
            raise FtdiDllError("Ftd2xx.dll을 찾을 수 없습니다. FTDI D2XX 드라이버가 설치되어 있는지 확인하세요.")
        if self.debug: print("[DEBUG] Ftd2xx.dll loaded successfully.")
        self._define_ftdi_functions()
        self._open_device_by_description(description_keyword)
        self.reader_thread = threading.Thread(target=self._read_data_loop, daemon=True)
        self.reader_thread.start()
        if self.debug: print("[DEBUG] Background reader thread started.")

    def _define_ftdi_functions(self):
        try:
            funcs = {'FT_ListDevices': ([c_void_p, c_void_p, c_ulong], c_ulong),
                     'FT_OpenEx': ([c_char_p, c_ulong, POINTER(c_void_p)], c_ulong),
                     'FT_Close': ([c_void_p], c_ulong), 'FT_ResetDevice': ([c_void_p], c_ulong),
                     'FT_Write': ([c_void_p, c_void_p, c_ulong, POINTER(c_ulong)], c_ulong),
                     'FT_Read': ([c_void_p, c_void_p, c_ulong, POINTER(c_ulong)], c_ulong),
                     'FT_GetQueueStatus': ([c_void_p, POINTER(c_ulong)], c_ulong),
                     'FT_Purge': ([c_void_p, c_ulong], c_ulong), 'FT_SetLatencyTimer': ([c_void_p, c_ubyte], c_ulong)}
            for name, (argtypes, restype) in funcs.items():
                func = getattr(self.ft_dll, name)
                func.argtypes, func.restype = argtypes, restype
                setattr(self, name.lower().replace('ft_', 'ft_'), func)
        except AttributeError as e:
            raise FtdiDllError(f"Ftd2xx.dll에서 함수를 찾을 수 없습니다: {e}")
        if self.debug: print("[DEBUG] All required FTDI functions are defined.")

    def _open_device_by_description(self, keyword: str):
        num_devs = c_ulong();
        status = self.ft_listdevices(byref(num_devs), None, FT_LIST_NUMBER_ONLY)
        if status != FT_OK or num_devs.value == 0: raise DeviceNotFoundError("연결된 FTDI 장치가 없습니다.")
        target_desc_bytes = None
        for i in range(num_devs.value):
            buffer = ctypes.create_string_buffer(64)
            if self.ft_listdevices(ctypes.c_void_p(i), buffer, FT_LIST_BY_INDEX | FT_OPEN_BY_DESCRIPTION) == FT_OK:
                desc_str = buffer.value.decode('latin-1', errors='ignore')
                if keyword.lower() in desc_str.lower(): target_desc_bytes = buffer.value; break
        if not target_desc_bytes: raise DeviceNotFoundError(f"설명에 '{keyword}' 키워드가 포함된 장치를 찾을 수 없습니다.")
        status = self.ft_openex(target_desc_bytes, FT_OPEN_BY_DESCRIPTION, byref(self.ft_handle))
        if status != FT_OK: raise LightBoxError(f"장치 열기 실패 (FT_OpenEx), status code: {status}")
        self.ft_resetdevice(self.ft_handle)
        time.sleep(0.1)
        latency_timer_ms = 2
        if self.ft_setlatencytimer(self.ft_handle, c_ubyte(latency_timer_ms)) == FT_OK:
            if self.debug: print(f"[DEBUG] Latency Timer set to {latency_timer_ms}ms.")
        self.ft_purge(self.ft_handle, FT_PURGE_RX | FT_PURGE_TX)
        self._is_open = True
        if self.debug: print("[DEBUG] Device opened and initialized successfully.")

    def _read_data_loop(self):
        while not self._stop_thread_event.is_set():
            if not self._is_open or not self.ft_handle: time.sleep(0.1); continue
            queue_status = c_ulong()
            if self.ft_getqueuestatus(self.ft_handle, byref(queue_status)) == FT_OK and queue_status.value > 0:
                bytes_read = c_ulong();
                read_data_buffer = ctypes.create_string_buffer(queue_status.value)
                if self.ft_read(self.ft_handle, read_data_buffer, queue_status.value, byref(bytes_read)) == FT_OK:
                    with self.rx_lock: self.rx_buffer.extend(read_data_buffer.raw[:bytes_read.value])
            time.sleep(0.005)

    def close(self):
        if self._is_open:
            if self.debug: print("[DEBUG] Closing device...")
            if self.ft_handle:
                try:
                    print("\n프로그램 종료를 위해 조명을 끕니다...")
                    self.set_light_level(0)  # 밝기 0으로 설정
                    self.power_control(False)  # 전원 끄기
                except LightBoxError as e:
                    if self.debug: print(f"[WARNING] Failed to turn off light on close: {e}")
            self._stop_thread_event.set()
            if self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=1.0)
            if self.ft_handle: self.ft_close(self.ft_handle); self.ft_handle = None
            self._is_open = False
            if self.debug: print("[DEBUG] Device closed successfully.")

    def send_command(self, command: int, data: int, timeout_ms: int = 300) -> int:
        if not self._is_open: raise LightBoxError("장치가 열려있지 않습니다.")
        with self.rx_lock:
            self.rx_buffer.clear()
        pkt = bytes([_STX1, _STX2, command, (data >> 16) & 0xFF, (data >> 8) & 0xFF, data & 0xFF, _END])
        if self.debug: print(f"[DEBUG] TX: CMD=0x{command:02X}, DATA=0x{data:06X} -> PKT={pkt.hex()}")
        bytes_written = c_ulong();
        status = self.ft_write(self.ft_handle, pkt, len(pkt), byref(bytes_written))
        if status != FT_OK or bytes_written.value != len(pkt): return -1002
        deadline = time.time() + timeout_ms / 1000.0
        while time.time() < deadline:
            with self.rx_lock:
                if len(self.rx_buffer) >= 3:
                    ack_data = self.rx_buffer
                    if self.debug: print(f"[DEBUG] RX: {ack_data.hex()}")
                    # ACK의 command가 보낸 command와 일치하는지 확인
                    if ack_data[0] == _STX1 and ack_data[1] == command and ack_data[2] == _END:
                        return 0
                    else:
                        return -1
            time.sleep(0.01)
        if self.debug: print(f"[DEBUG] ACK Timeout for CMD 0x{command:02X}.")
        return -1005

    # [새로운 함수] 전원 제어
    def power_control(self, turn_on: bool) -> bool:
        """조명 컨트롤러의 주 전원을 켜거나 끕니다."""
        power_data = 1 if turn_on else 0
        state_str = "ON" if turn_on else "OFF"
        print(f"\n1. 전원 {state_str} 명령 전송 (CMD: 0x{_CMD_POWER_ON:02X}, DATA: {power_data})...")
        ack = self.send_command(_CMD_POWER_ON, power_data)
        print(f"   -> 응답(ACK): {ack}")
        if ack != 0: print(f"[실패] 전원 {state_str}에 실패했습니다."); return False
        return True

    # [수정된 함수] 밝기 설정
    def set_light_level(self, value: int) -> bool:
        """조명 밝기를 설정합니다."""
        print(f"2. 밝기 설정 명령 전송 (CMD: 0x{_CMD_SET_BRIGHTNESS:02X}, DATA: {value})...")
        ack = self.send_command(_CMD_SET_BRIGHTNESS, value)
        print(f"   -> 응답(ACK): {ack}")
        if ack != 0: print("[실패] 밝기 설정에 실패했습니다."); return False
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ----------------------------
# 메인 실행 블록 (CLI)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="조명 컨트롤러 CLI (ZIG_COMMAND 최종 적용)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("level", type=int, nargs='?', default=1000, help="설정할 밝기 값 (기본값: 1000)")
    parser.add_argument("--desc", default="Camera-Tester", help="검색할 장치 설명 키워드")
    parser.add_argument("--debug", action="store_true", help="자세한 디버그 메시지 출력")
    args = parser.parse_args()

    try:
        with LightBoxController(description_keyword=args.desc, debug=args.debug) as lb_controller:
            # 1. 전원 켜기
            if not lb_controller.power_control(turn_on=True):
                raise LightBoxError("초기화 실패: 전원을 켤 수 없습니다.")

            # 2. 밝기 설정
            if not lb_controller.set_light_level(value=args.level):
                raise LightBoxError("밝기 설정에 실패했습니다.")

            print("\n[최종 성공] 조명이 켜졌습니다.")
            input("Enter 키를 누르면 프로그램을 종료합니다...")

    except (LightBoxError, FtdiDllError, DeviceNotFoundError) as e:
        print(f"\n[오류 발생] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[알 수 없는 오류 발생] {e}", file=sys.stderr)
        sys.exit(1)