# Snag Detection Demo (YOLO11n)

A small proof-of-concept object detection model that finds **wall cracks** (construction snags) in site photos. It was built to test whether a lightweight YOLO11n model can be trained on a laptop with no dedicated GPU and still work on images it has never seen.

> **Status:** proof of concept. The model is trained on a very small dataset and is meant to demonstrate the pipeline, not to be used in production.

---

## Table of contents
1. [Project goal](#project-goal)
2. [Results](#results)
3. [Repository structure](#repository-structure)
4. [Quick start](#quick-start)
5. [Dataset](#dataset)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [OpenVINO export and benchmark](#openvino-export-and-benchmark)
9. [Problems faced and how they were fixed](#problems-faced-and-how-they-were-fixed)
10. [Limitations](#limitations)
11. [Future work](#future-work)
12. [License and credits](#license-and-credits)

---

## Project goal

Construction snags (defects such as wall cracks) are normally found by manual inspection. The goal of this project is to check whether a small, fast detector can flag them automatically from photos, and later run offline on low-end Android phones as part of a site-inspection app.

This repository is the first step: train a small model, verify it works on unseen images, and document the process.

## Results

Validation results of the final model (`model/snag_detector.pt`, the best checkpoint of the 50-epoch run):

| Metric | Value |
|---|---|
| Precision | 0.48 |
| Recall | 0.50 |
| mAP50 | 0.48 |
| mAP50-95 | 0.22 |

**Important:** the validation set has only 7 images with 10 labelled objects, so these numbers are noisy. During training, mAP50 swung between about 0.01 and 0.56 from one epoch to the next. Treat them as a rough indication, not a precise measure.

The model was also tested on new site photos that were not part of the dataset, and it detected cracks correctly in the cases tested. See the `results/` folder for examples.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── predict.py              # run the model on an image or folder
├── .gitignore
├── model/
│   └── snag_detector.pt    # trained weights (best.pt from training)
├── samples/                # example input images
└── results/                # example outputs, training curves, confusion matrix
```

## Quick start

**1. Install**
```bash
pip install -r requirements.txt
```

**2. Run on an image or a folder**
```bash
python predict.py samples/your_image.jpg
python predict.py samples
```
Annotated images are saved to `results/output/`.

**Or in Python:**
```python
from ultralytics import YOLO

model = YOLO("model/snag_detector.pt")
results = model.predict(source="samples/your_image.jpg", imgsz=640, conf=0.25, save=True)
```

**Tuning detection:** raise `conf` (for example 0.4) if you get too many wrong boxes, and lower it (for example 0.15) if the model misses cracks.

## Dataset

- **Task:** object detection, 1 class (wall crack).
- **Source:** exported from Roboflow in YOLO format. Add the dataset name, link and license here: `[DATASET LINK AND LICENSE]`.
- **Split used:**

| Split | Images |
|---|---|
| Train | 32 |
| Validation | 7 |
| Test | 4 |

The dataset is not included in this repository. Please download it from the source above and respect its license.

Expected layout:
```
dataset/
├── data.yaml
├── train/images, train/labels
├── valid/images, valid/labels
└── test/images,  test/labels
```

## Training

- **Model:** YOLO11n (2.59 million parameters), starting from COCO-pretrained weights (transfer learning)
- **Framework:** Ultralytics 8.4.218, PyTorch 2.15 (CPU build)
- **Hardware:** Intel Core i5-1035G1 laptop, CPU only, 11.8 GB RAM
- **Training time:** about 14 minutes for 50 epochs

Command used:
```bash
yolo detect train data=dataset/data.yaml model=yolo11n.pt imgsz=640 epochs=50 batch=4 device=cpu workers=0
```

| Parameter | Value | Why |
|---|---|---|
| `model` | yolo11n.pt | Smallest YOLO11 model: fast, light on RAM, suits mobile deployment later |
| `imgsz` | 640 | Standard size the pretrained weights expect; multiple of 32 |
| `epochs` | 50 | 10 epochs was too few; metrics only started improving after about epoch 30 |
| `batch` | 4 | Fits in laptop RAM; about 15 seconds per epoch |
| `device` | cpu | No NVIDIA GPU available |
| `workers` | 0 | Avoids multiprocessing crashes and freezes on Windows |

Defaults used automatically: AdamW optimizer (learning rate 0.002 picked by `optimizer=auto`), mosaic augmentation (switched off for the last 10 epochs), horizontal flip, HSV colour changes, scale and translate augmentation.

## Evaluation

Check the model on your own images:
```bash
yolo detect predict model=model/snag_detector.pt source="path/to/images" imgsz=640 conf=0.25 save=True
```

Validate on the test split:
```bash
yolo detect val model=model/snag_detector.pt data=dataset/data.yaml split=test imgsz=640 device=cpu workers=0
```

Training curves and the confusion matrix are in `results/`. Training loss fell steadily (classification loss from 5.49 to 1.64), while box loss improved only slightly. This matches the low mAP50-95 and suggests box placement is still loose.

## OpenVINO export and benchmark

To check whether the model can run fast on modest hardware, YOLO11n was exported to OpenVINO and benchmarked on the laptop's Intel integrated GPU and on CPU:

```python
from ultralytics import YOLO
YOLO("model/snag_detector.pt").export(format="openvino", imgsz=640)
```

| Device | Average latency (30 runs) |
|---|---|
| Intel GPU (OpenVINO) | about 47 ms per image |
| CPU (OpenVINO) | about 59 ms per image |

This benchmark was run with the pretrained `yolo11n` on a sample image, so it shows hardware speed rather than the accuracy of the crack model. The Intel GPU is useful for **inference only**. OpenVINO cannot train models, so training stayed on CPU.

## Problems faced and how they were fixed

| Problem | Cause | Fix |
|---|---|---|
| Jupyter kernel crashed when training started; the terminal showed `OMP: Error #15: libiomp5md.dll already initialized` | Two copies of the OpenMP runtime (one from Anaconda's Intel MKL, one bundled with PyTorch) were loaded in the same process | Set `KMP_DUPLICATE_LIB_OK=TRUE` before running (`set KMP_DUPLICATE_LIB_OK=TRUE` in Anaconda Prompt, or `os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"` at the top of a notebook). This is a workaround, not a full fix; a clean Python 3.12 environment avoids the duplicate library |
| Earlier I suspected low RAM | Only about 2.6 GB of RAM was free, but the log showed the real cause was the OpenMP conflict | Confirmed by running training from the terminal, where the error message was visible |
| OpenVINO error: `Could not open the file yolo11n_openvino_model/model.xml` | Ultralytics names the exported file `yolo11n.xml`, not `model.xml` | Pointed to the correct file name |
| First GPU prediction appeared to take about 11 seconds | One-time model compilation and warm-up on the GPU | Benchmarked with warm-up runs; steady-state latency is about 47 ms |
| `imgsz=412` in the first config | 412 is not a multiple of 32, so it is silently rounded to 416 | Switched to `imgsz=640` |
| Very unstable validation metrics | Only 7 validation images (10 objects), so one hit or miss moves mAP a lot | Judged the model by visual tests on unseen images as well as the metrics |
| Wanted to train on the Intel GPU | OpenVINO supports inference only | Trained on CPU with `yolo11n`, which finished in about 14 minutes |

## Limitations

- **Very small dataset** (32 training images): the model may not generalise across different lighting, camera angles, wall textures and sites.
- **Loose boxes:** mAP50-95 of 0.22 shows box placement is not yet precise.
- **Noisy evaluation:** the validation set is too small for reliable metrics.
- **One class only:** wall cracks. Other snag types are not covered.
- **Not production-ready:** use as a demo only.

## Future work

- Collect and label more images (aim for a few hundred or more) across different sites and conditions.
- Retrain for more epochs, ideally on a GPU, and compare with other mobile-friendly detectors.
- Export for mobile (NCNN or LiteRT) and test offline on 4 GB and 6 GB Android phones.
- Add more snag classes.
- Generate inspection reports from the detections.

## License and credits

- Built with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), which is licensed under **AGPL-3.0**. Commercial use of the model may require an Ultralytics enterprise license.
- Dataset: `[DATASET NAME AND LICENSE]`.
- Author: Aryan
