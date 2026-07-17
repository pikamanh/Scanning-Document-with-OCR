# Extract Information ID Card

A desktop application for extracting information from the front side of a Vietnamese Citizen Identity Card (CCCD) using a webcam or RTSP stream. The system combines YOLO, CTPN, and VietOCR to detect the card, locate text lines, and recognize Vietnamese text in near real time.

> The current pipeline assumes a fixed front-side CCCD layout and is primarily configured for an NVIDIA GPU with CUDA support.

## Features

- Captures video from the default webcam or an RTSP URL.
- Detects the CCCD region using a trained YOLO model.
- Crops, rotates, and deskews the card before text recognition.
- Detects individual text lines with CTPN.
- Recognizes Vietnamese text with VietOCR.
- Runs inference in a worker thread to keep the PyQt6 interface responsive.
- Displays the extracted information in the desktop interface.
- Extracts the following fields:
  - ID number
  - Full name
  - Date of birth
  - Gender
  - Nationality
  - Place of origin
  - Place of residence (two lines)
  - Date of expiry
- Includes scripts for training the YOLO and CTPN models.

## Processing Pipeline

```mermaid
flowchart LR
    A[Webcam / RTSP] --> B[Read frames with OpenCV]
    B --> C[Detect the ID card with YOLO]
    C --> D[Crop and rotate 90 degrees]
    D --> E[Deskew with Hough Transform]
    E --> F[Detect text lines with CTPN]
    F --> G[Recognize text with VietOCR]
    G --> H[Normalize and validate fields]
    H --> I[Display results in PyQt6]
```

The interface reads a video frame approximately every 30 ms. By default, one frame is submitted to the recognition worker after every 30 captured frames, preventing the inference pipeline from blocking the UI. YOLO detections must have a confidence score of at least `0.8`.

## Technology Stack

- **Python**, **PyTorch**, and **Torchvision**
- **Ultralytics YOLO** for ID card detection
- **CTPN** (Connectionist Text Proposal Network) for text-line detection
- **VietOCR** with the `vgg_transformer` configuration for Vietnamese text recognition
- **OpenCV** for video capture, cropping, rotation, and image preprocessing
- **PyQt6** for the desktop interface and worker thread

## System Requirements

- Python 3.10 or 3.11.
- A webcam or an accessible RTSP stream.
- An NVIDIA GPU and CUDA 11.8 according to the current dependency configuration.
- An Internet connection on the first run if the pretrained VietOCR weights are not already available in the local cache.

Although some components support CPU inference, the application currently passes `device="cuda"` to both YOLO and VietOCR by default. Running the complete pipeline on a CPU requires changing the device configuration in `app/utils/threading.py` and `app/recognizer/OCR.py`.

## Installation

### 1. Clone the repository

Using SSH:

```bash
git clone git@github.com:pikamanh/Extract-Information-IDCard.git
cd Extract-Information-IDCard
```

Or using HTTPS:

```bash
git clone https://github.com/pikamanh/Extract-Information-IDCard.git
cd Extract-Information-IDCard
```

### 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current `requirements.txt` pins `torch==2.7.1` and `torchvision==0.22.1`, and includes the PyTorch package index for CUDA 11.8. If the machine uses a different CUDA version, install the matching PyTorch and Torchvision builds first, then install the remaining packages.

The YOLO training script also imports `rich`, which is not currently listed in `requirements.txt`:

```bash
python -m pip install rich
```

## Running the Application

The model and UI files are loaded through relative paths. Run the application from the **repository root**:

```bash
python main.py
```

Usage:

1. Select `Webcam` or `RTSP` in the setup dialog.
2. If RTSP is selected, enter the complete stream URL and click `OK`.
3. Click `Start Camera` in the main window.
4. Place the front side of the CCCD inside the frame. Keep the card well lit, clearly visible, and correctly oriented.
5. Once all fields pass validation, the application displays the extracted values and stops the camera.
6. Click `Stop Camera` to stop the stream manually.

Example RTSP URL:

```text
rtsp://username:password@192.168.1.100:554/stream
```

Do not commit an RTSP URL containing camera credentials to the source code.

## Output Schema

The OCR module returns a dictionary with the following structure:

```python
{
    "id_number": "001234567890",
    "name": "NGUYEN VAN A",
    "dob": "01/01/1990",
    "gender": "Nam",
    "national": "Việt Nam",
    "place_orgin": "...",
    "place_of_residence1": "...",
    "place_of_residence2": "...",
    "date_expired": "01/01/2030"
}
```

The `place_orgin` key intentionally preserves the spelling used by the current source code. Before accepting a result, the UI verifies that:

- `id_number` contains either 9 or 12 characters;
- `gender` is exactly `Nam` or `Nữ`;
- the date of birth and expiry date follow the `dd/mm/yyyy` format.

## Project Structure

```text
.
├── main.py                         # Entry point, UI, and camera control
├── requirements.txt                # Application dependencies
├── model/
│   ├── best_detection_cccd.pt      # YOLO weights for CCCD detection
│   ├── new 2.pth                   # CTPN weights used during inference
│   └── new.pth                     # An additional CTPN checkpoint
├── app/
│   ├── detector/
│   │   ├── predict.py              # CTPN inference and text-line assembly
│   │   ├── train.py                # CTPN training entry point
│   │   ├── ctpn/
│   │   │   ├── config.py           # Dataset, anchor, and training settings
│   │   │   ├── ctpn.py             # CTPN architecture and loss functions
│   │   │   ├── dataset.py          # Pascal VOC and ICDAR dataset loaders
│   │   │   └── utils.py            # Anchors, IoU, NMS, and text-line linking
│   │   ├── weights/                # Pretrained weights for resuming CTPN
│   │   └── log/                    # Training plots and example results
│   ├── recognizer/
│   │   └── OCR.py                  # ROI extraction, VietOCR, and postprocessing
│   ├── resources/
│   │   ├── main.ui                 # Main-window Qt UI definition
│   │   ├── setup.ui                # Video-source dialog definition
│   │   └── resource.qrc            # Qt resource declaration
│   └── utils/
│       ├── pre_proccessing.py      # Image deskewing
│       └── threading.py            # YOLO → preprocessing → OCR worker
└── trainning/
    └── train_yolo.py               # CLI for training or resuming YOLO
```

The following local/generated directories are listed in `.gitignore`:

- `datasets/`: training datasets.
- `runs/`: outputs generated by Ultralytics.
- `temp/`: temporary cropped images created during inference.
- `results/`: output images produced by standalone detector experiments.
- `app/detector/checkpoints/`: CTPN training checkpoints.

## Models in the Pipeline

| Model | File/configuration | Purpose |
| --- | --- | --- |
| YOLO | `model/best_detection_cccd.pt` | Detects the CCCD bounding box in a video frame |
| CTPN | `model/new 2.pth` | Detects text proposals and connects them into text lines |
| VietOCR | `vgg_transformer` configuration | Recognizes Vietnamese text in each detected line |

Replacement weights must use the format expected by the source code. YOLO requires an Ultralytics-compatible checkpoint, while the CTPN checkpoint must be a dictionary containing a `model_state_dict` key.

## Training YOLO

The `trainning/train_yolo.py` script discovers datasets using the following structure:

```text
datasets/
└── <dataset-name>/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

Run the training script from the repository root:

```bash
python trainning/train_yolo.py
```

The script prompts for:

1. A new training run (`no`) or a resumed run (any value other than `no`).
2. The dataset to use, selected by its displayed index.

Review the following details before training:

- The new-training branch currently uses `yolov8n-pose.yaml` and `yolov8n-pose.pt`. If the dataset only contains CCCD bounding boxes, replace them with a suitable object-detection model.
- The resume branch expects `model/best.pt`. Place the checkpoint at that location or update the path.
- The default configuration uses 200 epochs, an image size of `640`, and a patience value of 50.
- Ultralytics writes checkpoints and training outputs to `runs/` by default.

## Training CTPN

CTPN supports Pascal VOC XML annotations and also includes an ICDAR loader in `app/detector/ctpn/dataset.py`. The current training script uses `VOCDataset` with the following default layout:

```text
datasets/
├── image/                           # Training images
└── xml/                             # Pascal VOC annotations
```

Before training, review these values in `app/detector/ctpn/config.py`:

- `img_dir` and `label_dir`;
- `pretrained_weights`;
- `checkpoints_dir`;
- the worker count and anchor/IoU thresholds, if required.

On Linux and macOS, replace the default Windows-style paths containing `\` with valid paths. Create the checkpoint directory if it does not exist, then run the script from the repository root:

```bash
mkdir -p app/detector/checkpoints
python app/detector/train.py
```

The default training configuration uses a batch size of 1, SGD with a learning rate of `1e-3`, 80 epochs, and learning-rate milestones at epochs 35, 55, and 70. If a pretrained checkpoint is found, training resumes from the checkpoint's `epoch` value.

## Inference Configuration

Common parameters that may need adjustment:

| Parameter | Location | Default |
| --- | --- | --- |
| Frame interval | `Camera(..., skip_frame=30)` in `main.py` | 30 frames |
| YOLO confidence | `model_yolo.predict(..., conf=0.8)` | 0.8 |
| YOLO device | `CameraThread(device="cuda")` | CUDA |
| VietOCR device | `OCR(device="cuda")` | CUDA |
| CTPN confidence | `get_text_boxes(..., prob_thresh=0.5)` | 0.5 |
| CTPN weights | `weights` in `app/detector/predict.py` | `model/new 2.pth` |

## Current Limitations

- Field extraction depends on line order. CTPN sorts boxes by vertical position and maps them sequentially to nine fixed fields.
- Only text boxes with `y > 200` in the resized `640 × 640` image are passed to VietOCR.
- The cropped card is always rotated 90 degrees clockwise, so the input orientation must match this assumption.
- Deskewing uses Hough Lines and may fail on blurred, poorly lit images or images without clear edges.
- Only the first YOLO detection is processed when multiple cards appear in a frame.
- The project does not currently provide a REST API, batch-image mode, or automated tests. The camera interface is the primary workflow.
- Model and UI paths depend on the current working directory.

## Troubleshooting

### CUDA errors or `Torch not compiled with CUDA enabled`

Verify the installed NVIDIA driver, CUDA runtime, and PyTorch build. The source currently defaults to CUDA. For CPU inference, change both the `CameraThread` and `OCR` devices to `cpu`.

### The webcam or RTSP stream does not open

- Ensure the webcam is not in use by another application.
- Test the RTSP URL with VLC or FFmpeg.
- Check the firewall, credentials, camera codec, and network connectivity.

### `No detection`

- Keep the entire CCCD visible inside the frame.
- Improve lighting and reduce glare or motion blur.
- Consider lowering the YOLO confidence threshold from `0.8` if the detector repeatedly misses the card.

### Fields are missing or the application keeps asking to try again

The pipeline requires all expected text lines in the correct layout and valid output formats. Keep the front side of the card straight, clearly visible, correctly oriented, and unobstructed. Lowering the CTPN `prob_thresh` may recover missed lines, but a value that is too low can introduce noisy boxes and shift the field order.

## References

- [CTPN paper](https://arxiv.org/abs/1609.03605)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [VietOCR](https://github.com/pbcquoc/vietocr)
- The CTPN component is based on the implementations listed in `app/detector/README.md`.
