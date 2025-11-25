#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/core/camera_exceptions.py

카메라 제어 및 관련 작업에서 발생할 수 있는 예외 클래스들을 정의합니다.
"""

class CameraError(Exception):
    """카메라 관련 오류의 기본 예외 클래스."""
    pass

class CameraConnectionError(CameraError):
    """카메라 연결 실패 시 발생하는 예외."""
    pass

class CameraNotConnectedError(CameraError):
    """카메라가 연결되지 않은 상태에서 작업을 시도할 때 발생하는 예외."""
    pass

class GrabberError(CameraError):
    """이미지 캡처(grabber) 관련 오류의 기본 예외 클래스."""
    pass

class GrabberNotActiveError(GrabberError):
    """그래빙이 활성화되지 않은 상태에서 프레임을 요청할 때 발생하는 예외."""
    pass

class GrabberStartError(GrabberError):
    """그래빙 시작 실패 시 발생하는 예외."""
    pass

class GrabberStopError(GrabberError):
    """그래빙 중지 실패 시 발생하는 예외."""
    pass

class FrameTimeoutError(GrabberError):
    """프레임 획득 타임아웃 시 발생하는 예외."""
    pass

class FrameAcquisitionError(GrabberError):
    """프레임 획득 실패 시 발생하는 예외."""
    pass

class ParameterError(CameraError):
    """파라미터 관련 오류의 기본 예외 클래스."""
    pass

class ParameterSetError(ParameterError):
    """파라미터 설정 실패 시 발생하는 예외."""
    pass

class ParameterGetError(ParameterError):
    """파라미터 값 조회 실패 시 발생하는 예외."""
    pass

class CommandExecutionError(CameraError):
    """카메라 명령 실행 실패 시 발생하는 예외."""
    pass