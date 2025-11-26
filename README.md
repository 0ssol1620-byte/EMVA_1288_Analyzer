# EMVA 1288 Analyzer

**EMVA 1288 Analyzer** is a PyQt5-based GUI application that helps you measure and document
camera performance according to the **EMVA 1288** standard.

It coordinates:

- A CoaXPress camera (via Euresys eGrabber)
- A controllable lightbox (via FTDI)
- Automated capture sequences

and writes the final results into a ready-made **Excel template**.

The design goal is:

> To let test / validation engineers run full EMVA 1288 measurements  
> through a guided UI, with minimal scripting – while still offering a clean  
> Python implementation for custom workflows.

---

## Table of Contents

1. [What is EMVA 1288 Analyzer?](#what-is-emva-1288-analyzer)  
2. [Key Benefits](#key-benefits)  
3. [Typical Measurement Workflow](#typical-measurement-workflow)  
4. [High-level Architecture](#high-level-architecture-1)  
5. [UI Overview – Measuring Without Code](#ui-overview--measuring-without-code)  
   - [Connection & Lightbox Control](#connection--lightbox-control)  
   - [Measurement Panel](#measurement-panel)  
   - [Results & Excel Export](#results--excel-export)  
6. [Core Computation: EMVA Parameters](#core-computation-emva-parameters)  
7. [Requirements & Installation](#requirements--installation-1)  
8. [Running the Application](#running-the-application-1)  
9. [Notes & Limitations](#notes--limitations)

---

## What is EMVA 1288 Analyzer?

**EMVA 1288** is a standard that defines how to measure key camera parameters such as:

- Dark noise
- Signal-to-noise ratio (SNR)
- Responsivity
- Linearity
- Saturation capacity

The EMVA 1288 Analyzer application guides you through capturing the necessary frames,
computes the key metrics, and writes the results to a structured Excel template
(`Template_format.xlsx`) for reporting.

---

## Key Benefits

- ✅ **Guided workflow** – measurement steps are driven from the UI, not from scripts.  
- ✅ **Integrated control** – camera and lightbox are controlled from one application.  
- ✅ **Repeatability** – same configuration can be re-run across firmware or hardware variants.  
- ✅ **Standardized reporting** – outputs into a consistent Excel template that can be shared.  
- ✅ **Extensible** – the internals are written in Python (NumPy, PyQt5), easy to extend.

---

## Typical Measurement Workflow

1. **Setup**
   - Connect the camera to the Coaxlink frame grabber.
   - Connect the lightbox (via FTDI).
   - Ensure dark/flat-field conditions are available (optical setup).

2. **Connect & configure**
   - Launch the app and connect to the camera.
   - Verify basic operation (live view, exposure, gain).
   - Configure lightbox intensity ranges.

3. **Run EMVA sequence**
   - Choose EMVA measurement settings:
     - Bayer pattern
     - Exposure steps
     - Number of frames per step
     - Repeats for statistics
   - Start measurement and let the app:
     - Move through dark / illuminated conditions
     - Capture A/B frame pairs
     - Compute EMVA metrics

4. **Review & export**
   - Inspect summary results in the UI.
   - Export to `Template_format.xlsx` for archiving or sharing.

---

## High-level Architecture

```text
EMVA_1288_Analyzer/
├─ main.py                      # Main GUI and application entry point
├─ Template_format.xlsx         # Excel template for EMVA 1288 results
├─ vieworks.ico                 # Windows icon
├─ core/
│  ├─ camera_controller.py      # eGrabber-based camera control
│  ├─ camera_exceptions.py      # Camera-related exception definitions
│  ├─ camera_facade.py          # Qt-friendly CxpCamera wrapper
│  ├─ controller_pool.py        # CameraController pool management
│  ├─ emva_processor.py         # EMVA computation logic
│  └─ lightbox_controller.py    # FTDI-based lightbox control
└─ workers/
   ├─ dark_data_worker.py       # Dark-frame collection worker
   ├─ grab_worker.py            # Background frame-draining worker
   └─ measurement_worker.py     # EMVA measurement sequence worker
```

- **core/** – device control and EMVA math  
- **workers/** – background threads for draining frames and executing long-running measurements  
- **main.py** – ties everything together in a PyQt5 `QMainWindow`  

---

## UI Overview – Measuring Without Code

### Connection & Lightbox Control

The main window (`EMVAAnalyzerApp`) typically includes:

- **Camera connection panel**
  - Connect / reconnect camera.
  - Shows connection state and basic info.
- **Lightbox control panel**
  - On/Off switch.
  - Brightness setting in normalized units (`0 .. DEFAULT_LB_MAX_UNITS`).
  - Communication status (e.g. FTDI link OK).

For a non-programmer:

> “Connect the camera, turn on the lightbox, choose brightness – all from UI buttons and sliders.”

---

### Measurement Panel

This is the core of the GUI for EMVA runs.

Common controls:

- **Bayer pattern**
  - Select how the raw image is organized (e.g. RGGB, GRBG, BGGR…).
  - Ensures correct channel separation in `emva_processor.py`.

- **Exposure & gain settings**
  - Base exposure time, or a range over which to sweep.
  - Optional gain configuration.

- **Frame counts**
  - How many frames to capture per step.
  - How many repeats to perform for better estimates of noise.

- **Buttons**
  - **Start measurement** – kicks off `MeasurementWorker`.
  - **Stop** – requests the worker to stop gracefully.

- **Progress display**
  - Progress bar showing current step vs total.
  - Log text or status messages indicating:
    - Current illumination / exposure.
    - Which step is being processed (dark, light, saturation, etc.).

The idea is to make the EMVA run feel like a **wizard**:

1. Adjust settings.
2. Start measurement.
3. Wait until progress completes.

---

### Results & Excel Export

After measurement:

- The UI can display:
  - Summary metrics:
    - Mean signal / noise vs illumination.
    - Key EMVA parameters derived from those values.
  - Per-channel results (for Bayer sensors).

- **Excel Export**
  - The app:
    1. Locates a `Template_format.xlsx` (template candidates).
    2. Copies the template to an output file.
    3. Uses `openpyxl` to write:
       - Measured points
       - Derived EMVA parameters
       - Any required metadata (camera ID, measurement conditions, etc.).

The result is an Excel file that can be directly used for:

- Internal reports
- Data comparisons
- External sharing with customers or partners

---

## Core Computation: EMVA Parameters

All core math lives in `core/emva_processor.py`.

### Channels

- **Bayer RAW**
  - `separate_bayer_channels` splits into Gr, R, Gb, B.
- **RGB**
  - Standard channel separation.

### A/B frame pairs

Most 1288-style measurements require **pairs of frames** under the same condition:

- Frame A and Frame B at the same illumination and exposure.
- Per-pixel operations:
  - Mean signal: `(A + B) / 2`
  - Temporal noise: `std(A − B) / sqrt(2)`

The module provides functions to:

- Compute mean, temporal noise, and total noise.
- Aggregate per-channel and per-illumination-level results.
- Prepare data for insertion into the template.

While the full EMVA 1288 spec can be complex, the goal here is to
encapsulate the **most common calculations** needed for internal testing.

---

## Requirements & Installation

### Python & OS

- **Python**: 3.8+ recommended  
- **OS**: Windows (Euresys Coaxlink + eGrabber)  

### Python dependencies

Key packages:

- `PyQt5` – GUI
- `numpy` – numeric operations
- `opencv-python` – for optional display/preview scaling
- `openpyxl` – Excel reading/writing for `Template_format.xlsx`
- `egrabber` – camera SDK

Example `requirements.txt`:

```text
PyQt5==5.15.11
numpy==1.24.4
opencv-python==4.11.0.86
openpyxl==3.1.5
egrabber==25.3.2.80
```

> Make sure the Euresys SDK installer has added the Python bindings and DLLs to
> the appropriate locations.

---

## Running the Application

From the `EMVA_1288_Analyzer` directory:

```bash
pip install -r requirements.txt
python main.py --theme auto
```

Possible options (depending on your version):

- `--theme auto|light|dark` – UI theme.
- `--icon path/to/icon.ico` – override window icon.

If packaged with **PyInstaller**, `main.py` includes logic to handle:

- The PyInstaller runtime directory (`sys._MEIPASS`).
- Location of `Template_format.xlsx` and resource files.

---

## Notes & Limitations

- The app assumes an EMVA-like workflow; it is not a general-purpose sequencer
  like ViewTS, though the internal structure is modular.
- Hardware-accurate EMVA measurements still depend heavily on:
  - Optical setup (uniform illumination, stable lightbox).
  - Correct dark/shutter handling.
  - Stable environmental conditions.

However, by standardizing:

- How images are captured  
- How statistics are computed  
- How results are written to Excel  

the EMVA 1288 Analyzer significantly reduces the manual labor and potential
for mistakes in day-to-day EMVA testing.

