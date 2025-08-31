# DataLoaders

A `DataLoader` is a crucial utility that wraps a `Dataset` and provides an iterable over it. Its primary responsibilities are to handle batching, shuffling, and multi-process data loading, ensuring that the GPU is fed with data efficiently without becoming a bottleneck.

While LibTorch provides a basic data loading API (`torch::data::DataLoader`), it can be complex to use and lacks some of the convenient features found in Python's `torch.utils.data.DataLoader`.

xTorch simplifies and enhances this process with its own high-performance implementations.

## `xt::dataloaders::ExtendedDataLoader`

The `ExtendedDataLoader` is the primary, high-level data loader in xTorch. It is designed to be both easy to use and highly performant, mirroring the functionality and simplicity of its Python counterpart.

It abstracts away the complexity of parallel data fetching and provides a simple `for` loop interface for iterating over batches of data.

### Key Features

-   **Simple API**: Requires minimal setup and configuration.
-   **Multi-Process Data Loading**: Uses multiple worker processes to load data in parallel, preventing CPU bottlenecks.
-   **Automatic Batching**: Combines individual data samples into batches.
-   **Optional Shuffling**: Can automatically shuffle the data at the beginning of each epoch.
-   **Prefetching**: Pre-fetches batches in the background to keep the GPU saturated.

### Usage

The `ExtendedDataLoader` is typically initialized with a dataset object and configuration options. It can then be used in a range-based `for` loop to retrieve data batches.

```cpp
#include <xtorch/xtorch.h>

int main() {
    // 1. Assume 'dataset' is an initialized xt::datasets::Dataset object
    auto dataset = xt::datasets::MNIST("./data", xt::datasets::DataMode::TRAIN);

    // 2. Instantiate the ExtendedDataLoader
    xt::dataloaders::ExtendedDataLoader data_loader(
        dataset,
        /*batch_size=*/64,
        /*shuffle=*/true,
        /*num_workers=*/4,
        /*prefetch_factor=*/2
    );

    // 3. Iterate over the data loader to get batches
    torch::Device device(torch::kCUDA);
    int batch_count = 0;
    for (auto& batch : data_loader) {
        // Each 'batch' is a pair of (data, target) tensors
        torch::Tensor data = batch.first.to(device);
        torch::Tensor target = batch.second.to(device);

        if (batch_count == 0) {
            std::cout << "Batch Data Shape: " << data.sizes() << std::endl;
            std::cout << "Batch Target Shape: " << target.sizes() << std::endl;
        }
        batch_count++;
    }
    std::cout << "Total batches: " << batch_count << std::endl;
}
```

### Constructor Parameters

The `ExtendedDataLoader` is configured through its constructor:

`ExtendedDataLoader(Dataset& dataset, size_t batch_size, bool shuffle = false, int num_workers = 0, int prefetch_factor = 2)`

| Parameter | Type | Description |
|---|---|---|
| `dataset` | `xt::datasets::Dataset&` | The dataset from which to load the data. |
| `batch_size` | `size_t` | The number of samples per batch. |
| `shuffle` | `bool` | If `true`, the data is reshuffled at every epoch. Defaults to `false`. |
| `num_workers`| `int` | The number of subprocesses to use for data loading. `0` means that the data will be loaded in the main process. Defaults to `0`. |
| `prefetch_factor`| `int`| The number of batches to prefetch in advance for each worker. This helps hide data loading latency. Defaults to `2`. |

---

## Integration with `xt::Trainer`

The `ExtendedDataLoader` is designed to work seamlessly with the `xt::Trainer`. You simply pass your initialized data loader instances to the `trainer.fit()` method.

```cpp
// Assume model, optimizer, train_loader, and val_loader are initialized
xt::Trainer trainer;
trainer.set_max_epochs(10)
       .set_optimizer(optimizer)
       .set_loss_fn(torch::nll_loss);

// The trainer will automatically iterate over the data loaders
trainer.fit(model, train_loader, &val_loader, device);
```

For most use cases, the `xt::dataloaders::ExtendedDataLoader` is the recommended and only data loader you will need.
