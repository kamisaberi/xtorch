# 📊 Development Progress Tracker

## Status Key
| Badge | Status         | Description                          |
|-------|----------------|--------------------------------------|
| ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | **Finished** | Completed and merged. |
| ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | **In Progress** | Actively being developed. |
| ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | **Under Review** | Pending review/testing. |
| ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | **Not Started** | Not yet implemented. |
| ![On Hold](https://img.shields.io/badge/-On_Hold-yellow) | **On Hold** | Paused (optional). |

---

## Section: Datasets
| Component       | Status Badge | Notes |
|----------------|-------------|-------|
| Authentication | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
| Database Layer | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
| API Docs | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
| Caching | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |

---

## Section: Models
| Component       | Status Badge | Notes |
|----------------|-------------|-------|
| Authentication | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
| Database Layer | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
| API Docs | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
| Caching | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |

---

## Section: Transforms : Image
| Section               | Component           | Status | Notes |
|-----------------------|---------------------|-------|-------|
| **BASIC**             |                     | | |
|                       | Resize              | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | CenterCrop          | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|                       | RandomCrop          | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
|                       | Pad                 | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
|                       | RandomHorizontalFlip | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | RandomVerticalFlip  | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | RandomFlip          | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | RandomRotation      | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | RandomResizedCrop   | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
| **Color/Contrast**    |                     | | |
|                       | ColorJitter         | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | Grayscale           | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|                       | RandomGrayscale     | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
|                       | RandomInvert        | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
|                       | RandomAdjustSharpness | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | RandomAutoContrast  | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | RandomEqualize      | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
| **Geometry**          |                     | | |
|                       | RandomAffine         | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | ElasticTransform           | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|                       | RandomPerspective     | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
|                       | RandomThinPlateSpline           | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|                       | RandomShadow      | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
| **Tensor Operations** |                     | | |
|                       | ToTensor          | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | Normalize           | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|                       | ConvertImageDtype     | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
|                       | Lambda         | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
| **Augmentation**      | | | |
|                       | RandomApply | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | RandomChoice | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|                       | RandomOrder | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
|                       | AutoAugment | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
|                       | RandAugment | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | MixUp | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | CutMix | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|                       | GridMask | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
|                       | CutOut  | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
|                       | StyleGAN-based augmentations | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
| **Advanced**          | | | |
|                       | GaussianBlur | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                       | RandomSolarize | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|                       | RandomPosterize | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
| **Domain Specific**   | | | |
|                       | Weather Effects | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |



---

## Section: Transforms : Signal
| Section         | Component           | Status | Notes |
|-----------------|---------------------|-------|-------|
| **Preprocessing**      |                     | | |
|                 | Resample              | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                 | Vol          | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|                 | SlidingWindowCmn          | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
| **Spectrograms**    |                     | | |
|                 | Spectrogram         | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                 | MelSpectrogram           | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|                 | MFCC     | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
|                 | GriffinLim        | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
| **Time-Frequency**    |                     | | |
|                 | TimeStretch         | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                 | FrequencyMasking           | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|                 | TimeMasking     | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
|                 | InverseMelScale           | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|                 | MelScale      | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
| **Augmentation** |                     | | |
|                 | SpeedPerturbation          | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                 | AddNoise            | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|                 | PitchShift      | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
|                 | BackgroundNoiseAddition          | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
|                 | TimeWarping           | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
| **Advanced** | | | |
|                 | De-reverberation | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|                 | WaveletTransforms  | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |


---

## Section: Transforms : Video
| Section    | Component           | Status | Notes |
|------------|---------------------|-------|-------|
| **Frame-Based** |                     | | |
|            | Resample              | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|            | Vol          | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|            | SlidingWindowCmn          | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
| **Temporal** |                     | | |
|            | Spectrogram         | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|            | MelSpectrogram           | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|            | MFCC     | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
|            | GriffinLim        | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
| **3D Augmentation**  |                     | | |
|            | TimeStretch         | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|            | FrequencyMasking           | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|            | TimeMasking     | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
|            | InverseMelScale           | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|            | MelScale      | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
| **Motion-Based** |                     | | |
|            | SpeedPerturbation          | ![Finished](https://img.shields.io/badge/-Finished-brightgreen) | Stable in v1.0. |
|            | AddNoise            | ![In Progress](https://img.shields.io/badge/-In_Progress-orange) | Adding NoSQL support. |
|            | PitchShift      | ![Under Review](https://img.shields.io/badge/-Under_Review-blue) | PR #56 open. |
|            | BackgroundNoiseAddition          | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
|            | TimeWarping           | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |

---


### How to Use
1. Replace `Section: Core Module` with your actual module name.
2. Update badges dynamically by changing the status in the URL (e.g., `In_Progress` → `Finished`).
3. Optional: Add direct links to PRs/issues in the **Notes** column.