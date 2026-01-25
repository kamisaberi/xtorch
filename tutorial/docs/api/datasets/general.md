# General Purpose Datasets

While xTorch provides handlers for many public datasets, the most common use case is training a model on your own custom data. To make this process as simple as possible, xTorch provides a set of general-purpose dataset classes designed to work with common data formats and directory structures.

These classes save you from writing a custom C++ `Dataset` class from scratch for many standard scenarios. They are all located under the `xt::datasets` namespace and can be found in the `<xtorch/datasets/general/>` header directory.

---

## `xt::datasets::ImageFolderDataset`

This is one of the most useful dataset classes in the library. `ImageFolderDataset` allows you to load a custom image classification dataset from a directory, provided it follows a specific structure.

### Required Directory Structure

The data must be organized into a root folder, with one subdirectory for each class. Each subdirectory should contain all the images belonging to that class.

```
/path/to/your/data/
├── class_a/
│   ├── xxx.png
│   ├── xxy.png
│   └── ...
│
├── class_b/
│   ├── zzz.png
│   ├── zzy.png
│   └── ...
│
└── class_c/
├── 123.png
├── 456.png
└── ...
```

The dataset will automatically discover the classes based on the subdirectory names and assign an integer label to each class.

### Usage

You instantiate `ImageFolderDataset` with the path to the root data folder and an optional transform pipeline. It handles all the file discovery and label assignment internally.

```cpp
#include <xtorch/xtorch.h>

int main() {
    // 1. Define image transformations
    auto transforms = std::make_unique<xt::transforms::Compose>(
        std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{224, 224}),
        std::make_shared<xt::transforms::general::Normalize>(
            std::vector<float>{0.5, 0.5, 0.5},
            std::vector<float>{0.5, 0.5, 0.5}
        )
    );

    // 2. Instantiate the ImageFolderDataset with the path to the root directory
    auto my_dataset = xt::datasets::ImageFolderDataset(
        "/path/to/your/data/", // Path to the root folder shown above
        std::move(transforms)
    );

    std::cout << "Found " << *my_dataset.size() << " images in "
              << my_dataset.classes().size() << " classes." << std::endl;

    // 3. Pass to a DataLoader as usual
    xt::dataloaders::ExtendedDataLoader data_loader(my_dataset, 32, true);

    // Ready for training
    for (auto& batch : data_loader) {
        // ...
    }
}
```

---

## `xt::datasets::CSVDataset`

This class provides a simple way to load datasets from a `.csv` file. It is highly flexible, allowing you to specify which columns contain features and which column contains the target label.

### Usage

```cpp
#include <xtorch/xtorch.h>

int main() {
    // Assume you have a CSV file "my_data.csv" with columns:
    // feature_1, feature_2, feature_3, target_class

    // Specify the names of the columns to be used as features
    std::vector<std::string> feature_cols = {"feature_1", "feature_2", "feature_3"};

    // Specify the name of the column to be used as the target
    std::string target_col = "target_class";

    // Instantiate the CSVDataset
    auto csv_dataset = xt::datasets::CSVDataset(
        "path/to/my_data.csv",
        feature_cols,
        target_col
    );

    xt::dataloaders::ExtendedDataLoader data_loader(csv_dataset, 16);

    for (auto& batch : data_loader) {
        auto features = batch.first;
        auto targets = batch.second;
        // ... training step ...
    }
}
```

---

## `xt::datasets::TensorDataset`

This is a utility dataset that wraps one or more existing tensors. It is useful when your entire dataset already fits in memory as `torch::Tensor` objects. Each sample will be a slice along the first dimension of the given tensors.

### Usage

```cpp
// Create some random data and targets
auto all_features = torch::randn({1000, 20}); // 1000 samples, 20 features each
auto all_targets = torch::randint(0, 5, {1000}); // 1000 target labels

// Wrap the tensors in a TensorDataset
auto tensor_dataset = xt::datasets::TensorDataset({all_features, all_targets});

std::cout << "TensorDataset size: " << *tensor_dataset.size() << std::endl; // Prints 1000

xt::dataloaders::ExtendedDataLoader data_loader(tensor_dataset, 100);
```

---

## Other General-Purpose Handlers

xTorch includes several other `Folder`-style datasets for different data modalities.

| Dataset Class | Description | Header File |
|---|---|---|
| `AudioFolder` | Loads audio files from a directory structure similar to `ImageFolder`. | `general/audio_folder.h` |
| `TextFolder` | Loads text files from a directory structure similar to `ImageFolder`. | `general/text_folder.h` |
| `VideoFolder` | Loads video files from a directory structure similar to `ImageFolder`. | `general/video_folder.h` |
| `PairedImageDataset`| Loads paired images (e.g., for style transfer or image-to-image translation) from two corresponding directories. | `general/paired_image_dataset.h` |
