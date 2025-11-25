# EMVA 1288 Analyzer

PyQt5 기반 GUI 도구로, 카메라와 라이트박스를 제어하면서 **EMVA 1288 규격에 따른 카메라 특성 측정**을 쉽게 할 수 있도록 만든 애플리케이션입니다.

- Coaxlink + eGrabber 기반 카메라 제어
- FTDI 기반 라이트박스(LightBox) 밝기 제어
- Dark / Light / Saturation 등의 시퀀스를 자동으로 돌려서 EMVA용 데이터를 수집
- 수집된 데이터를 **Template_format.xlsx** 템플릿에 자동으로 써 넣어 결과 정리

---

## 주요 기능

### 1. 카메라 제어 (CoaXPress / eGrabber)

`core/camera_controller.py`, `core/camera_facade.py`, `core/controller_pool.py` 모듈이 담당합니다.

- Euresys **eGrabber** SDK 를 이용해 카메라 검색 / 연결 / 해제 / 프레임 그래빙
- 싱글/멀티 카메라를 모두 지원할 수 있도록 Controller 풀 구조 사용
- 프레임은 `numpy.ndarray` (2D/3D) 형태로 상위 계층에 전달
- PyQt5 신호/슬롯과 연동하기 위해 `CxpCamera(QObject)` 래퍼 제공

### 2. 라이트박스 제어

`core/lightbox_controller.py` 에서 FTDI DLL 을 `ctypes` 로 직접 호출하여 라이트박스를 제어합니다.

- FT_ListDevices / FT_OpenEx 등 저수준 API 래핑
- 밝기 세트, ON/OFF, 통신 모니터링 등을 포함
- 카메라 측 노출/게인과 연동하여 EMVA 측정 시 요구되는 조도 레벨을 재현

### 3. EMVA 1288 데이터 처리

`core/emva_processor.py` 는 EMVA 계산에 필요한 기초 연산을 제공합니다.

- Bayer RAW:
  - Gr, R, Gb, B 채널 분리 (`separate_bayer_channels`)
- RGB 이미지:
  - R, G, B 채널 분리
- 두 장의 이미지(예: 동일 조건에서 찍은 A/B 프레임)에 대해
  - 채널별 평균 신호 (mean)
  - 임시 노이즈(temporal noise)
  - 총 노이즈(total noise)
  를 계산하는 `calculate_emva_parameters` 제공

MeasurementWorker 는 이 함수를 호출해서, 측정 시퀀스에서 얻은 프레임 쌍들을 EMVA 규격에 맞는 숫자로 변환합니다.

### 4. 측정 워커

`workers/measurement_worker.py`, `workers/dark_data_worker.py`, `workers/grab_worker.py` 가 각각 역할을 분담합니다.

- **GrabWorker**
  - 백그라운드 QThread 로 카메라 프레임 큐를 꾸준히 비우는 역할
  - 필요 시 `cv2.resize` 를 사용해 표시용으로 다운스케일
- **MeasurementWorker**
  - 카메라 + 라이트박스를 함께 제어하면서,
  - 노출/조도 단계별로 여러 프레임을 캡처
  - EMVA용 평균/노이즈 계산을 수행하고 결과를 UI와 엑셀로 전달
- **DarkDataWorker**
  - 다크(Shutter closed) 상태에서 평균/노이즈를 안정적으로 쌓기 위한 별도 워커

---

## 디렉터리 구조

```text
EMVA_1288_Analyzer/
├─ main.py                      # 메인 GUI 및 앱 엔트리 포인트
├─ Template_format.xlsx         # EMVA 1288 결과를 기록할 엑셀 템플릿
├─ vieworks.ico                 # 윈도우 아이콘
├─ core/
│  ├─ camera_controller.py      # eGrabber 기반 카메라 제어
│  ├─ camera_exceptions.py      # 카메라 관련 예외 정의
│  ├─ camera_facade.py          # Qt 용 CxpCamera 래퍼
│  ├─ controller_pool.py        # CameraController 풀 관리
│  ├─ emva_processor.py         # EMVA 값 계산 로직
│  └─ lightbox_controller.py    # FTDI 기반 라이트박스 제어
└─ workers/
   ├─ dark_data_worker.py       # 다크 데이터 수집용 워커
   ├─ grab_worker.py            # 백그라운드 프레임 드레이닝 워커
   └─ measurement_worker.py     # EMVA 측정 시퀀스 워커
```

---

## 아키텍처 다이어그램 (Mermaid)

GitHub 에서 바로 렌더링되도록 Mermaid 다이어그램을 넣어두면 구조 파악에 도움이 됩니다.

```mermaid
flowchart LR
    subgraph HW[하드웨어]
        Cam[Camera (CoaXPress)]
        LB[LightBox (FTDI)]
    end

    subgraph Core
        CC[camera_controller.py]
        CF[camera_facade.py<br/>CxpCamera]
        CP[controller_pool.py]
        LBCTL[lightbox_controller.py]
        EP[emva_processor.py]
    end

    subgraph Workers
        GW[grab_worker.py]
        MW[measurement_worker.py]
        DDW[dark_data_worker.py]
    end

    subgraph UI
        APP[EMVAAnalyzerApp (main.py)]
        XLS[Template_format.xlsx<br/>(openpyxl)]
    end

    Cam --> CC --> CF
    CF --> GW
    CF --> MW
    CF --> DDW

    LB --> LBCTL --> MW

    MW --> EP
    MW --> APP
    DDW --> APP

    MW --> XLS
    APP --> XLS
```

---

## 메인 UI (main.py)

`EMVAAnalyzerApp(QMainWindow)` 가 전체 UI를 구성합니다.

- Connection 영역
  - 카메라 연결 상태 표시 (Status: Disconnected / Connected)
  - Reconnect 버튼
- LightBox 제어 영역
  - 밝기 단위 (0 ~ DEFAULT_LB_MAX_UNITS) 설정
  - 온/오프 및 통신 상태 표시
- Measurement 패널
  - Bayer 패턴(예: RGGB, GRBG 등) 선택
  - 노출 시간, 프레임 수, 반복 횟수 등 입력
  - 측정 시작 / 중단 버튼
  - 진행률(ProgressBar) 및 현재 스텝 메시지 표시
- 결과 표/그래프 영역
  - 각 단계별 mean / noise 결과를 테이블 형태로 표시
  - 필요 시 Template_format.xlsx 로 내보내기

엑셀 저장 시에는:

1. `_template_candidates()` 로 템플릿 위치 후보를 찾고  
2. `Template_format.xlsx` 를 복사한 뒤  
3. `openpyxl` 로 복사본을 열어 측정 결과를 채워 넣습니다.

---

## 의존 패키지

코드에서 직접 사용하는 외부 패키지는 다음과 같습니다.

- `PyQt5`  : GUI (Qt Widgets, Core, Gui)
- `numpy`  : 수치 계산 및 프레임 데이터 처리
- `opencv-python` (`cv2`) : GrabWorker 에서 표시용 리사이즈에 사용
- `openpyxl` : `Template_format.xlsx` 엑셀 템플릿 읽기/쓰기
- `egrabber` : Euresys Coaxlink / eGrabber 카메라 SDK (Python 바인딩)

---

## requirements.txt 예시

이 프로젝트를 위한 최소 requirements 는 아래와 같이 정리할 수 있습니다
(당신이 제공한 환경의 버전을 기준으로 고정).

```text
PyQt5==5.15.11
numpy==1.24.4
opencv-python==4.11.0.86
openpyxl==3.1.5
egrabber==25.3.2.80
```

> PyQt5-Qt5, PyQt5_sip 등은 `PyQt5` 설치 시 함께 따라오므로 별도로 적지 않아도 됩니다.

---

## 설치 및 실행

```bash
pip install -r requirements.txt
```

### 일반 실행

```bash
python main.py --theme auto
```

옵션 예시:

- `--theme auto|light|dark` : 테마 선택
- `--icon path/to/icon.ico` : 윈도우 아이콘 강제
- PyInstaller 로 패키징된 exe 환경에서도 동작하도록,
  `_template_candidates()` 와 아이콘 경로 로직이 PyInstaller 런타임 폴더(`sys._MEIPASS`)를 우선적으로 참조하도록 구성되어 있습니다.

---

## 주의 사항

- 실제 카메라 / 라이트박스를 사용하려면
  - Euresys Coaxlink 보드 + eGrabber SDK 설치
  - LightBox 가 FTDI 기반으로 동작하는 환경에서 FTDI 드라이버가 설치되어 있어야 합니다.
- 하드웨어가 없더라도
  - 코드 일부는 더미 모드로 동작할 수 있지만,
  - EMVA 측정 본연의 기능은 실제 하드웨어 환경이 있어야 의미 있는 결과를 얻을 수 있습니다.
