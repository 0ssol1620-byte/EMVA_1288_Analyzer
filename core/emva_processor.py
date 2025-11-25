# core/emva_processor.py

import numpy as np
from typing import Union, Dict, Tuple

def separate_bayer_channels(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Bayer 이미지를 Gr, R, Gb, B 채널로 분리합니다. (BayerRG 패턴 가정)
    입력: 2D numpy 배열
    출력: 각 채널(2D 배열)을 값으로 하는 딕셔너리
    """
    if image.ndim != 2:
        raise ValueError(f"Bayer image must be 2D, but got {image.ndim} dimensions.")
    r = image[0::2, 0::2]
    gr = image[0::2, 1::2]
    gb = image[1::2, 0::2]
    b = image[1::2, 1::2]
    return {'R': r, 'Gr': gr, 'Gb': gb, 'B': b}

def separate_rgb_channels(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    RGB 이미지를 R, G, B 채널로 분리합니다.
    입력: 3D numpy 배열 (H, W, 3)
    출력: 각 채널(2D 배열)을 값으로 하는 딕셔너리
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Input must be an RGB image with shape (H, W, 3), but got shape {image.shape}")
    return {'R': image[..., 0], 'G': image[..., 1], 'B': image[..., 2]}

def calculate_emva_parameters(
        image_a: np.ndarray,
        image_b: np.ndarray,
        is_bayer: bool = False,
        is_rgb: bool = False
) -> Union[Tuple[float, float, float], Dict[str, Tuple[float, float, float]]]:
    """
    두 개의 연속된 이미지로부터 EMVA 1288 주요 파라미터(평균, 시간적 표준편차, 전체 표준편차)를 계산합니다.

    - Mono: (mean, std_temporal, std_total) 튜플을 반환합니다.
    - Bayer: 각 채널(Gr, R, Gb, B)별 결과를 담은 딕셔너리를 반환합니다.
    - RGB: 각 채널(R, G, B)별 결과를 담은 딕셔너리를 반환합니다.

    Args:
        image_a (np.ndarray): 첫 번째 이미지.
        image_b (np.ndarray): 두 번째 이미지.
        is_bayer (bool): Bayer 포맷인지 여부.
        is_rgb (bool): RGB 포맷인지 여부.

    Returns:
        Union[Tuple[float, float, float], Dict[str, Tuple[float, float, float]]]: 계산된 결과.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(f"The two images must have the same shape, but got {image_a.shape} and {image_b.shape}.")

    if is_bayer:
        channels_a = separate_bayer_channels(image_a)
        channels_b = separate_bayer_channels(image_b)
        results = {}
        for ch in ['Gr', 'R', 'Gb', 'B']:
            # 재귀 호출 시 is_bayer와 is_rgb를 False로 설정하여 Mono 로직 실행
            mean, std_temp, std_total = calculate_emva_parameters(
                channels_a[ch], channels_b[ch], is_bayer=False, is_rgb=False
            )
            results[ch] = (mean, std_temp, std_total)
        return results

    elif is_rgb:
        channels_a = separate_rgb_channels(image_a)
        channels_b = separate_rgb_channels(image_b)
        results = {}
        for ch in ['R', 'G', 'B']:
            # 재귀 호출 시 is_bayer와 is_rgb를 False로 설정하여 Mono 로직 실행
            mean, std_temp, std_total = calculate_emva_parameters(
                channels_a[ch], channels_b[ch], is_bayer=False, is_rgb=False
            )
            results[ch] = (mean, std_temp, std_total)
        return results

    else:
        # Mono 및 분리된 채널에 대한 계산 로직
        # 정밀한 계산을 위해 float64로 변환
        image_a_f = image_a.astype(np.float64)
        image_b_f = image_b.astype(np.float64)

        # 평균 신호 (uy)
        mean_signal = np.mean(image_b_f)

        # 전체 노이즈 (Total Noise, 공간+시간 노이즈)의 표준편차
        std_total = np.std(image_b_f)

        # 시간 노이즈 (Temporal Noise) 계산
        # Kt = ((a - b)^2) / 2
        diff_image = np.subtract(image_a_f, image_b_f)
        squared_diff = np.square(diff_image)
        mean_of_squared_diff = np.mean(squared_diff)
        variance_temporal = mean_of_squared_diff / 2.0
        std_temporal = np.sqrt(variance_temporal)

        return mean_signal, std_temporal, std_total