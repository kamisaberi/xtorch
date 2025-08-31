# Utilities

The `xt::utils` namespace provides a collection of standalone helper functions and classes for common tasks that frequently appear in deep learning workflows.

These utilities are designed to reduce boilerplate and simplify operations related to file handling, data conversion, and model inference.

---

## Model Serialization and Inference

These functions streamline the process of saving, loading, and running inference with models.

### `xt::load_model(const std::string& model_path)`
Loads a TorchScript model from a file.

-   **Parameters:**
    -   `model_path`: The file path to the serialized `.pt` or `.pth` model.
-   **Returns:** A `torch::jit::script::Module` ready for inference.

### `xt::utils::predict(torch::jit::script::Module& model, torch::Tensor& tensor)`
Performs a forward pass on a loaded TorchScript model.

-   **Parameters:**
    -   `model`: The loaded TorchScript module.
    -   `tensor`: The input tensor for the model.
-   **Returns:** An `at::Tensor` containing the model's output logits.

### `xt::utils::argmax(const at::Tensor& tensor)`
Finds the index of the maximum value in a 1D tensor. This is commonly used to get the predicted class index from the output logits.

-   **Parameters:**
    -   `tensor`: A 1D tensor of scores or probabilities.
-   **Returns:** An `int` representing the index of the highest value.

#### Example Inference Pipeline

```cpp
#include <xtorch/xtorch.hh>

int main() {
    // 1. Load the exported TorchScript model
    auto model = xt::load_model("resnet18_script.pt");

    // 2. Load and preprocess an image
    auto tensor = xt::utils::image_to_tensor("input.jpg");

    // 3. Get model predictions
    auto output_logits = xt::utils::predict(model, tensor);

    // 4. Find the class with the highest score
    int predicted_class = xt::utils::argmax(output_logits);

    std::cout << "Predicted class = " << predicted_class << std::endl;
}
```

---

## Image Handling

These utilities simplify the process of reading image files and converting them into tensors suitable for model input.

### `xt::utils::image_to_tensor(const std::string& image_path)`
Reads an image from a file path and converts it into a `torch::Tensor`.

-   **Details:** The function uses OpenCV internally. It reads the image, converts it to a `CHW` (Channels, Height, Width) tensor of type `kFloat`, and normalizes pixel values to the `[0, 1]` range. It does **not** add a batch dimension.
-   **Parameters:**
    -   `image_path`: The file path to the image (`.jpg`, `.png`, etc.).
-   **Returns:** A 3D `torch::Tensor` with the shape `[Channels, Height, Width]`.

---

## Filesystem and Data Handling

These utilities help manage files and directories, which is especially useful when downloading datasets or handling file paths.

### `xt::utils::downloader(const std::string& url, const std::string& dest_path)`
Downloads a file from a URL to a specified destination.

### `xt::utils::extract(const std::string& archive_path, const std::string& dest_dir)`
Extracts a compressed archive (e.g., `.zip`, `.tar.gz`) to a destination directory.

### `xt::utils::mkdir(const std::string& dir_path)`
Creates a directory if it does not already exist.

### `xt::utils::path_join(const std::string& p1, const std::string& p2)`
Joins two path components with the correct platform-specific separator.

### `xt::utils::md5(const std::string& file_path)`
Computes the MD5 checksum of a file, which is useful for verifying the integrity of downloaded datasets.
