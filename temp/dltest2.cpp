// Include necessary LibTorch headers (Torch C++ API).
// <torch/torch.h> covers most of the C++ front-end (including datasets and DataLoader utilities).
#include <torch/torch.h>
// If using TorchVision C++ for ImageFolder, include TorchVision headers as well:
// (Uncomment the line below if TorchVision is available and ImageFolder is provided by it)
// #include <torchvision/vision.h>

#include <iostream>
#include <vector>
#include <optional>

// Define a custom DataLoader by inheriting from torch::data::DataLoaderBase.
// Template parameters:
//   Dataset - the dataset type (e.g., torch::data::datasets::ImageFolder)
//   Batch   - the type of batch to return (here we use the default dataset batch type, a vector of examples)
//   BatchRequest - the type representing a request for a batch (here a vector of indices).
template <typename Dataset>
class CustomDataLoader : public torch::data::DataLoaderBase<Dataset, typename Dataset::BatchType, std::vector<size_t>> {
  using Base = torch::data::DataLoaderBase<Dataset, typename Dataset::BatchType, std::vector<size_t>>;

public:
  // Constructor: takes a dataset object (moved in) and DataLoaderOptions for configuration.
  CustomDataLoader(Dataset dataset, torch::data::DataLoaderOptions options)
      : Base(options), sampler_(dataset.size().value_or(0)) {
    // This loader is single-threaded, so enforce 0 worker threads:
    if (this->options_.workers != 0) {
      throw std::runtime_error("CustomDataLoader: workers must be 0 for single-threaded loading.");
    }
    // When workers==0, DataLoaderBase uses main_thread_dataset_ to load data synchronously on the main thread&#8203;:contentReference[oaicite:0]{index=0}.
    // Move the provided dataset into the DataLoaderBase's internal storage.
    this->main_thread_dataset_ = std::make_unique<Dataset>(std::move(dataset));
  }

  // Destructor: ensure any internal threads (if used) are joined and resources are cleaned up.
  ~CustomDataLoader() {
    this->join();  // In single-thread mode, this will ensure all data has been processed.
  }

protected:
  // Sampler to generate sequential indices for each batch. We use a SequentialSampler internally.
  torch::data::samplers::SequentialSampler sampler_;

  // Override get_batch_request() to yield the next set of indices for a batch&#8203;:contentReference[oaicite:1]{index=1}.
  std::optional<std::vector<size_t>> get_batch_request() override {
    // Get the next batch of indices from the sequential sampler (size = batch_size).
    auto indices = sampler_.next(this->options_.batch_size);
    if (!indices.has_value()) {
      // No more indices available (dataset exhausted).
      return std::nullopt;
    }
    // If the sampler returns a smaller batch (end of dataset) and drop_last is true, skip this incomplete batch.
    if (indices->size() < this->options_.batch_size && this->options_.drop_last) {
      return std::nullopt;
    }
    // Return the batch indices to load.
    return indices;
  }

  // Override reset() to reset the sampler and the base DataLoader state&#8203;:contentReference[oaicite:2]{index=2}.
  void reset() override {
    sampler_.reset();            // Reset sampler to restart from beginning.
    Base::reset();               // Reset DataLoaderBase (clears internal counters, etc.).
  }

public:
  // Method to fetch the next batch of data as a single Example containing a batch of images and labels.
  // Returns an optional Example. When the dataset is exhausted, returns std::nullopt.
  std::optional<torch::data::Example<>> next_batch() {
    // Use DataLoaderBase::next() to get the next batch from the dataset (synchronously on main thread)&#8203;:contentReference[oaicite:3]{index=3}.
    // This returns an optional BatchType (here, a vector of Example<>) containing all samples in the batch.
    std::optional<typename Dataset::BatchType> batch = Base::next();
    if (!batch.has_value()) {
      // No more data (all batches have been consumed).
      return std::nullopt;
    }

    // The batch we got is a vector of examples (image tensor and label tensor for each sample).
    // Now stack the individual image tensors and label tensors into single batch tensors.
    const std::vector<torch::data::Example<>>& examples = *batch;  // reference to the vector of samples
    size_t batch_size = examples.size();
    std::vector<torch::Tensor> image_tensors;
    std::vector<torch::Tensor> label_tensors;
    image_tensors.reserve(batch_size);
    label_tensors.reserve(batch_size);

    for (const auto& example : examples) {
      // `example.data` is the image tensor, `example.target` is the label tensor.
      image_tensors.push_back(example.data);
      label_tensors.push_back(example.target);
    }

    // Stack all image tensors along a new 0th dimension (creating a [batch_size, ...] tensor).
    torch::Tensor images_batch = torch::stack(image_tensors);
    // Stack all label tensors (for scalar labels, this will create a 1D tensor of size [batch_size]).
    torch::Tensor labels_batch = torch::stack(label_tensors);

    // Return a single Example containing the batch of images and batch of labels.
    return torch::data::Example<>{images_batch, labels_batch};
  }
};

// Example usage of the CustomDataLoader with an ImageFolder dataset in main().
int main() {
  // Path to the image dataset directory (organized in subfolders for each class).
  std::string image_folder_path = "<your_image_dataset_path>";

  // 1. Instantiate the ImageFolder dataset (assumes images are arranged in class subdirectories).
  // Optionally, you might want to apply transformations (e.g., ToTensor) if ImageFolder returns image data in a different format.
  auto dataset = torch::data::datasets::ImageFolder(image_folder_path);
  // (If needed, apply transformations: e.g., dataset = dataset.map(torch::data::transforms::Normalize<>(...)) )

  // 2. Create the custom data loader with the dataset, specifying batch size and options.
  // Here we use batch_size = 4 and single-threaded loading (workers = 0).
  CustomDataLoader<decltype(dataset)> loader(dataset, torch::data::DataLoaderOptions().batch_size(4).workers(0));

  // 3. Iterate over the dataset using the custom loader.
  // We use a loop that calls next_batch() until the dataset is exhausted.
  size_t batch_index = 0;
  while (true) {
    auto batch = loader.next_batch();
    if (!batch.has_value()) {
      // No more batches available.
      break;
    }

    // Each `batch` is a torch::data::Example<> where:
    //   batch->data  is a tensor containing a batch of images (shape: [batch_size, channels, height, width])
    //   batch->target is a tensor containing a batch of labels (shape: [batch_size])
    std::cout << "Batch " << batch_index++ << " - "
              << "Loaded " << batch->data.size(0) << " images, "
              << "Batch image tensor size: " << batch->data.sizes() << ", "
              << "Batch label tensor size: " << batch->target.sizes() << std::endl;

    // Example usage: access the first image and label of the batch
    torch::Tensor first_image = batch->data[0];
    torch::Tensor first_label = batch->target[0];
    std::cout << "  First label in batch: " << first_label.item<int>() << std::endl;

    // (You can now feed `batch->data` and `batch->target` into a model for training or inference)
  }

  return 0;
}
