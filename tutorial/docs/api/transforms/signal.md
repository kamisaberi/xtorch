# Signal (Audio) Transforms

Signal transforms are a critical part of any audio-based deep learning pipeline. They are used to convert raw audio waveforms into representations that are more suitable for neural networks, and to perform data augmentation to improve model robustness.

Common tasks include converting a time-domain waveform into a time-frequency representation (like a spectrogram) and applying augmentations like pitch shifting or adding background noise.

All signal transforms are located under the `xt::transforms::signal` namespace and can be found in the `<xtorch/transforms/signal/>` header directory.

## General Usage

Audio transforms are designed to be chained together in a `Compose` pipeline and passed to an audio `Dataset`. The dataset will then apply this pipeline to each raw audio waveform it loads.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    // 1. Define a pipeline of audio transformations
    // This pipeline converts a raw waveform to a Mel Spectrogram and then applies augmentation.
    auto training_transforms = std::make_unique<xt::transforms::Compose>(
        // Convert the waveform to a Mel Spectrogram
        std::make_shared<xt::transforms::signal::MelSpectrogram>(
            /*sample_rate=*/16000,
            /*n_fft=*/400,
            /*n_mels=*/128
        ),
        // Apply Frequency Masking for data augmentation
        std::make_shared<xt::transforms::signal::FrequencyMasking>(
            /*freq_mask_param=*/80
        ),
        // Apply Time Masking for data augmentation
        std::make_shared<xt::transforms::signal::TimeMasking>(
            /*time_mask_param=*/100
        )
    );

    // 2. Pass the pipeline to an audio Dataset
    // The SpeechCommands dataset will now apply these transforms to each audio clip.
    auto dataset = xt::datasets::SpeechCommands(
        "./data",
        xt::datasets::DataMode::TRAIN,
        /*download=*/true,
        std::move(training_transforms)
    );

    // The data loader will yield batches of augmented spectrograms
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 32);
    // ...
}
```

!!! info "Constructor Options"
    Audio transforms are often highly configurable with parameters like `sample_rate`, `n_fft` (FFT window size), `hop_length`, and `n_mels` (number of Mel bands). Always refer to the specific header file in `<xtorch/transforms/signal/>` for a full list of available settings.

---

## Available Transforms by Category

### Time-Frequency Representations

These are the most common preprocessing steps, converting a 1D waveform into a 2D image-like representation.

| Transform | Description | Header File |
|---|---|---|
| `Spectrogram` | Creates a standard spectrogram from a waveform. | `spectrogram.h` |
| `MelSpectrogram`| Creates a Mel-scaled spectrogram, which is a perceptually relevant representation of audio. | `mel_spectrogram.h`|
| `MFCC` | Mel-Frequency Cepstral Coefficients, a compact representation of the spectral envelope. | `mfcc.h` |
| `InverseMelScale`| Converts a Mel-spectrogram to a regular spectrogram. | `inverse_mel_scale.h` |
| `GriffinLim` | An algorithm to estimate a waveform from a spectrogram (phase reconstruction). | `griffin_lim.h` |

### Data Augmentation

These transforms modify the audio to create new training samples, improving model generalization.

| Transform | Description | Header File |
|---|---|---|
| `TimeMasking` | Randomly masks a range of consecutive time steps in a spectrogram. A key component of SpecAugment. | `time_masking.h` |
| `FrequencyMasking`| Randomly masks a range of consecutive frequency channels in a spectrogram. A key component of SpecAugment. | `frequency_masking.h`|
| `AddNoise` | Adds random noise to the audio waveform. | `add_noise.h` |
| `BackgroundNoiseAddition`| Mixes the audio with random clips from a provided set of background noise files. | `background_noise_addition.h` |
| `PitchShift` | Shifts the pitch of the audio up or down without changing the tempo. | `pitch_shift.h` |
| `TimeStretch` | Stretches or compresses the audio in time without changing the pitch. | `time_stretch.h` |
| `SpeedPerturbation`| Changes the speed of the audio, which affects both pitch and tempo. Commonly used in ASR. | `speed_perturbation.h` |
| `DeReverberation`| Applies a de-reverberation effect to the audio. | `de_reverberation.h` |
| `TimeWarping` | Applies a non-linear warp along the time axis of a spectrogram. | `time_warping.h` |
| `Vol` | Changes the volume of the audio. | `vol.h` |

### Other Utility Transforms

| Transform | Description | Header File |
|---|---|---|
| `Resample` | Resamples the audio waveform from one sampling rate to another. | `resample.h` |
| `SlidingWindowCMN`| Cepstral Mean and Variance Normalization, a common technique in speech recognition. | `sliding_window_cmn.h`|
| `WaveletTransforms`| Applies wavelet transforms to the signal. | `wavelet_transforms.h` |
