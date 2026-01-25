# Transforms

Transforms are a fundamental component of any deep learning data pipeline. They are functions that take in a data sample (e.g., an image, a piece of text, or an audio clip) and return a modified version of it.

This process is essential for two primary reasons:
1.  **Preprocessing**: To convert data into a format that the neural network can accept. This includes resizing images to a fixed size, normalizing pixel values, or converting text tokens into numerical IDs.
2.  **Data Augmentation**: To artificially increase the diversity of the training dataset by applying random transformations (like random rotations or flips). This is a powerful regularization technique that helps the model generalize better to unseen data.

xTorch provides an extensive library of transforms for a wide variety of data types, mirroring and significantly extending the functionality found in popular Python libraries like `torchvision.transforms`.

## `xt::transforms::Compose`

A single transform performs one operation. To build a complete preprocessing or augmentation pipeline, you need to chain multiple transforms together. This is the job of `xt::transforms::Compose`.

`Compose` takes a list of transform modules and applies them sequentially to the data.

## General Usage

The standard workflow is to create a `Compose` object containing a list of the desired transform instances. This `Compose` object is then passed to a `Dataset` during its construction. The dataset will automatically apply this pipeline to each data sample it retrieves.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    // --- 1. Create a list of transform instances ---
    // This pipeline performs common data augmentation for image classification.
    std::vector<std::shared_ptr<xt::Module>> transform_list;
    transform_list.push_back(
        std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{256, 256})
    );
    transform_list.push_back(
        std::make_shared<xt::transforms::image::RandomCrop>(std::vector<int64_t>{224, 224})
    );
    transform_list.push_back(
        std::make_shared<xt::transforms::image::RandomHorizontalFlip>(/*p=*/0.5)
    );
    transform_list.push_back(
        std::make_shared<xt::transforms::general::Normalize>(
            std::vector<float>{0.485, 0.456, 0.406}, // Mean for ImageNet
            std::vector<float>{0.229, 0.224, 0.225}  // Std for ImageNet
        )
    );

    // --- 2. Create the Compose object ---
    auto transform_pipeline = std::make_unique<xt::transforms::Compose>(transform_list);

    // --- 3. Pass the pipeline to a Dataset ---
    // The ImageFolderDataset will now apply these augmentations to every image it loads.
    auto dataset = xt::datasets::ImageFolderDataset(
        "/path/to/your/image/data/",
        std::move(transform_pipeline)
    );

    // The rest of the workflow (DataLoader, Trainer) remains the same.
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 32);
    // ...
}
```

---

## Transforms by Data Modality

The xTorch transform library is organized by the type of data it operates on. Follow the links below for a detailed list of available transforms in each category.

-   **[Appliers](appliers.md)**: Meta-transforms that control how other transforms are applied, such as `RandomApply` or `OneOf`.

-   **[Image](image.md)**: The largest collection, containing transforms for geometric adjustments (resize, crop, rotate, flip), color jittering, normalization, and advanced augmentations like Cutout and MixUp.

-   **[Signal (Audio)](signal.md)**: Transforms for processing audio waveforms, including creating spectrograms (`MelSpectrogram`), applying time and frequency masking, and changing pitch or speed.

-   **[Text](text.md)**: Transforms for NLP, including tokenizers (`BertTokenizer`, `SentencePieceTokenizer`), and utilities for padding and truncating sequences.

-   **[Graph](graph.md)**: Transforms for augmenting graph-structured data, such as dropping nodes (`NodeDrop`) or edges (`EdgeDrop`).

-   **[Video](video.md)**: Transforms for processing sequences of images, such as temporal subsampling.

-   **[Weather](weather.md)**: A unique collection of transforms that can simulate various weather conditions (rain, fog, snow) on images, useful for training robust autonomous driving models.
