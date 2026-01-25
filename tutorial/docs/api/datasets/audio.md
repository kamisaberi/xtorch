# Audio & Speech Datasets

xTorch provides a rich collection of built-in dataset handlers for a wide range of audio and speech processing tasks, from recognition and classification to synthesis and source separation.

All audio datasets are located under the `xt::datasets` namespace and can be found in the `<xtorch/datasets/audio_processing/>` header directory.

## General Usage

The workflow for using an audio dataset is similar to that of other domains. You typically instantiate the dataset class with the path to your data and an optional pipeline of audio-specific transformations.

```cpp
#include <xtorch/xtorch.h>

int main() {
    // 1. Define audio-specific transformations (optional)
    // For example, creating a Mel Spectrogram from the raw audio waveform.
    auto transforms = std::make_unique<xt::transforms::Compose>(
        std::make_shared<xt::transforms::signal::MelSpectrogram>()
    );

    // 2. Instantiate a dataset for the Speech Commands task
    // The dataset will handle downloading and loading the data.
    auto dataset = xt::datasets::SpeechCommands(
        "./data",
        xt::datasets::DataMode::TRAIN,
        /*download=*/true,
        std::move(transforms)
    );

    std::cout << "Speech Commands dataset size: " << *dataset.size() << std::endl;

    // 3. Pass the dataset to a DataLoader
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 32, true);

    // The data loader is now ready for use in a training loop
    for (auto& batch : data_loader) {
        auto spectrograms = batch.first;
        auto labels = batch.second;
        // ... training step ...
    }
}
```

!!! info "Dataset Constructors"
Most dataset constructors follow a standard pattern:
`DatasetName(const std::string& root, DataMode mode, bool download, TransformPtr transforms)`
- `root`: The directory where the data is stored or will be downloaded.
- `mode`: `DataMode::TRAIN`, `DataMode::TEST`, or `DataMode::VALIDATION`.
- `download`: If `true`, the dataset will be downloaded if not found in the root directory.
- `transforms`: A `unique_ptr` to a transform pipeline to be applied to the data.

---

## Available Datasets by Task

### Audio Event Detection
| Dataset Class | Description | Header File |
|---|---|---|
| `AudioSet` | A large-scale dataset of manually annotated audio events from YouTube videos. | `audio_event_detection/audioset.h` |

### Binary Speech Classification
| Dataset Class | Description | Header File |
|---|---|---|
| `YesNo` | A small dataset of speech recordings saying "yes" or "no". | `binary_speech_classification/yes_no.h`|

### Emotion Recognition
| Dataset Class | Description | Header File |
|---|---|---|
| `IEMOCAP` | Interactive Emotional Dyadic Motion Capture database for emotion analysis. | `emotion_recognition/iemocap.h` |

### Environmental Sound Classification
| Dataset Class | Description | Header File |
|---|---|---|
| `ESC` | Dataset for Environmental Sound Classification (ESC-50 and ESC-10). | `environmental_sound_classification/esc.h` |
| `UrbanSound` | The UrbanSound8K dataset, containing urban sound recordings. | `environmental_sound_classification/urban_sound.h` |

### Intent Classification
| Dataset Class | Description | Header File |
|---|---|---|
| `Snips` | The Snips Natural Language Understanding benchmark dataset. | `intent_classification/snips.h` |

### Music Genre Classification
| Dataset Class | Description | Header File |
|---|---|---|
| `GTZAN` | A popular dataset for music genre recognition. | `music_genre_classification/gtzan.h` |

### Music Information Retrieval
| Dataset Class | Description | Header File |
|---|---|---|
| `MillionSongDataset` | A freely-available collection of audio features and metadata for a million contemporary popular music tracks. | `music_information_retrieval/million_song_dataset.h` |

### Music Source Separation
| Dataset Class | Description | Header File |
|---|---|---|
| `MUSDBHQ` | A high-quality dataset for music source separation. | `music_source_separation/mus_db_hq.h` |

### Music Tagging
| Dataset Class | Description | Header File |
|---|---|---|
| `MagnaTagATune` | A dataset of audio clips with associated tags. | `music_tagging/magna_tag_a_tune.h` |

### Sound Event Detection
| Dataset Class | Description | Header File |
|---|---|---|
| `FSD50K` | An open dataset of human-labeled sound events. | `sound_event_detection/fsd50k.h` |

### Speaker Identification and Verification
| Dataset Class | Description | Header File |
|---|---|---|
| `VoxCeleb` | An audio-visual dataset consisting of short clips of human speech, extracted from interview videos uploaded to YouTube. | `speaker_identification_and_verification/vox_celeb.h` |

### Speech Command Recognition
| Dataset Class | Description | Header File |
|---|---|---|
| `FluentSpeechCommands`| An audio dataset for spoken language understanding. | `speech_command_recognition/fluent_speech_commands.h` |
| `SpeechCommands` | A dataset of one-second audio clips of people saying thirty-five different words. | `speech_command_recognition/speech_commands.h` |

### Speech Recognition
| Dataset Class | Description | Header File |
|---|---|---|
| `CommonVoice` | A large, multi-language dataset of transcribed speech. | `speech_recognition/common_voice.h` |
| `LibriSpeech` | A corpus of read English speech suitable for training and evaluating speech recognition systems. | `speech_recognition/librispeech.h` |
| `TEDLIUM` | Audio recordings of TED talks with transcriptions. | `speech_recognition/tedlium.h` |
| `TIMIT` | A corpus of phonemically and lexically transcribed speech of American English speakers. | `speech_recognition/timit.h` |

### Speech Separation
| Dataset Class | Description | Header File |
|---|---|---|
| `LibriMix` | A dataset for source separation derived from LibriSpeech. | `speech_separation/libri_mix.h` |

### Speech Synthesis
| Dataset Class | Description | Header File |
|---|---|---|
| `CMUArctic` | Speech synthesis databases from Carnegie Mellon University. | `speech_synthesis/cmu_arctic.h` |
| `VCTK092` | Speech data uttered by 110 English speakers with various accents. | `speech_synthesis/vctk_092.h` |
| `LJSpeech` | A public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books. | `speech_synthesis/lj_speech.h` |
| `LibriTTS` | A large-scale, multi-speaker English corpus designed for TTS research. | `speech_synthesis/libritts.h` |
