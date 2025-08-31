# Datasets

The `Dataset` classes are responsible for accessing and preprocessing individual samples of data from a source (like a directory of images or a text file). They are the foundation of the data loading pipeline in xTorch.

A `Dataset` object is passed to a `DataLoader`, which then handles the more complex logic of batching, shuffling, and parallel data fetching.

## The `xt::datasets::Dataset` Base Class

All dataset classes in xTorch, both built-in and custom, inherit from a common base class: `xt::datasets::Dataset`. This class establishes a standard interface that the `DataLoader` knows how to interact with.

The two core methods of any dataset are:
-   `get(size_t index)`: Returns the data sample at the given index. This is typically a `torch::data::Example<>` containing a data tensor and a target tensor.
-   `size()`: Returns the total number of samples in the dataset as an `optional<size_t>`.

## Standard Usage

The typical workflow involves three steps:
1.  **Define Transforms**: Create a pipeline of data augmentation or preprocessing steps (optional).
2.  **Instantiate a Dataset**: Create an instance of a specific dataset class, providing the path to the data and the transform pipeline.
3.  **Pass to a DataLoader**: Pass the initialized dataset object to the `ExtendedDataLoader`.

```cpp
#include <xtorch/xtorch.h>

int main() {
    // 1. Define a transform pipeline
    auto compose = std::make_unique<xt::transforms::Compose>(
        std::make_shared<xt::transforms::general::Normalize>(0.5, 0.5)
    );

    // 2. Instantiate a built-in dataset for CIFAR-10
    auto cifar_dataset = xt::datasets::CIFAR10(
        "./data",
        xt::datasets::DataMode::TRAIN,
        /*download=*/true,
        std::move(compose)
    );

    // The dataset can report its size
    std::cout << "CIFAR-10 training set size: " << *cifar_dataset.size() << std::endl;

    // 3. Pass the dataset to a data loader
    xt::dataloaders::ExtendedDataLoader data_loader(cifar_dataset, 64, true);

    // The data loader can now be used for training
    for (auto& batch : data_loader) {
        // ...
    }
}
```

---

## Built-in Datasets by Domain

xTorch provides an extensive collection of built-in dataset handlers for many popular public datasets, saving you the effort of writing boilerplate loading code. These are organized by machine learning domain.

Follow the links below for a detailed list of available datasets in each category.

-   **[Computer Vision](computer-vision.md)**: Includes datasets for image classification (`MNIST`, `CIFAR`, `ImageNet`), object detection (`COCO`), and segmentation (`Cityscapes`).

-   **[Natural Language Processing](nlp.md)**: Includes datasets for text classification (`IMDB`, `AG_NEWS`), question answering (`SQuAD`), and machine translation (`WMT`).

-   **[Audio Processing](audio.md)**: Includes datasets for speech recognition (`LibriSpeech`), sound event classification (`UrbanSound8K`), and music analysis.

-   **[Time Series](time-series.md)**: Includes datasets for time series forecasting and classification.

-   **[Tabular Data](tabular.md)**: Includes a variety of classic small-scale datasets for classification and regression tasks.

-   **[Graph Data](graph.md)**: Includes datasets for node and graph classification tasks (`Cora`).

-   **[General Purpose](general.md)**: Contains flexible dataset classes like `ImageFolderDataset` and `CSVDataset` that allow you to easily load your own custom data without writing a new dataset class from scratch.

-   **[Other Domains](other.md)**: Includes datasets from other domains like biomedical data and recommendation systems.
