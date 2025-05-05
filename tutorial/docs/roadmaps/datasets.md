# Datasets Roadmap


## Datasets : CNN
| Section               | Component           | Status | Notes |
|-----------------------|---------------------|-------|-------|
| **LeNet**             |                     | | |
|                       | LeNet5              | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
| **AlexNet**    |                     | | |
|                       | AlexNet         | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
| **VGGNet**          |                     | | |
|                       | VGGNet11         | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|                       | VGGNet16           | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
| **ResNet** |                     | | |
|                       | ResNet18          | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
| **GoogleNet/Inception**      | | | |
|                       | InceptionV1 | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
| **UNet**          | | | |
|                       | UNet | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
| **MobileNet**   | | | |
|                       | MobileNet | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |

---

## Datasets : RNN
| Section         | Component           | Status | Notes |
|-----------------|---------------------|-------|-------|
| **Preprocessing**      |                     | | |
|                 | Resample              | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|                 | Vol          | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|                 | SlidingWindowCmn          | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
| **Spectrograms**    |                     | | |
|                 | Spectrogram         | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|                 | MelSpectrogram           | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|                 | MFCC     | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
|                 | GriffinLim        | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
| **Time-Frequency**    |                     | | |
|                 | TimeStretch         | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|                 | FrequencyMasking           | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|                 | TimeMasking     | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
|                 | InverseMelScale           | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|                 | MelScale      | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
| **Augmentation** |                     | | |
|                 | SpeedPerturbation          | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|                 | AddNoise            | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|                 | PitchShift      | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
|                 | BackgroundNoiseAddition          | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
|                 | TimeWarping           | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
| **Advanced** | | | |
|                 | De-reverberation | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|                 | WaveletTransforms  | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |

---

## Datasets : GAN
| Section    | Component           | Status | Notes |
|------------|---------------------|-------|-------|
| **Frame-Based** |                     | | |
|            | Resample              | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|            | Vol          | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|            | SlidingWindowCmn          | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
| **Temporal** |                     | | |
|            | Spectrogram         | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|            | MelSpectrogram           | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|            | MFCC     | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
|            | GriffinLim        | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
| **3D Augmentation**  |                     | | |
|            | TimeStretch         | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|            | FrequencyMasking           | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|            | TimeMasking     | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
|            | InverseMelScale           | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|            | MelScale      | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
| **Motion-Based** |                     | | |
|            | SpeedPerturbation          | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|            | AddNoise            | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|            | PitchShift      | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
|            | BackgroundNoiseAddition          | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
|            | TimeWarping           | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |

---

## Datasets : DIFFUSION
| Section    | Component           | Status | Notes |
|------------|---------------------|-------|-------|
| **Basic** |                     | | |
|            | Resample              | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|            | Vol          | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|            | SlidingWindowCmn          | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
| **Tokenization** |                     | | |
|            | Spectrogram         | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|            | MelSpectrogram           | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|            | MFCC     | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
|            | GriffinLim        | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
| **Sequencing**  |                     | | |
|            | TimeStretch         | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|            | FrequencyMasking           | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|            | TimeMasking     | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
|            | InverseMelScale           | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|            | MelScale      | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
| **Advanced** |                     | | |
|            | SpeedPerturbation          | ![Finished](https://img.shields.io/badge/-Finished-black) | Stable in v1.0. |
|            | AddNoise            | ![In Progress](https://img.shields.io/badge/-In_Progress-darkgrey) | Adding NoSQL support. |
|            | PitchShift      | ![Under Review](https://img.shields.io/badge/-Under_Review-grey) | PR #56 open. |
|            | BackgroundNoiseAddition          | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |
|            | TimeWarping           | ![Not Started](https://img.shields.io/badge/-Not_Started-lightgrey) | Planned for v2. |

---