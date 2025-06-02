#include <iostream>
#include <torch/torch.h>
#include <torch/data/datasets.h> // For torch::data::Dataset
#include <torch/data/example.h>  // For torch::data::Example
#include <vector>
#include <string>
#include <algorithm> // For std::shuffle, std::iota, std::min, std::max
#include <random>    // For std::default_random_engine, std::random_device
#include <iostream>
#include <thread>
#include <future>    // Not directly used in V2, but good to keep in mind for other patterns
#include <mutex>
#include <condition_variable>
#include <deque>      // For the batch queue
#include <atomic>     // For atomic counters and flags
#include <optional>   // For torch::optional and std::optional
#include <stdexcept>  // For std::out_of_range, std::runtime_error, std::invalid_argument
#include <chrono>     // For timing in main
#include "include/data_loaders/prefetch_data_loader.h"

#include "../../include/datasets/computer_vision/image_classification/mnist.h"

using namespace std;






int main()
{
    std::cout.precision(10);

    // xt::transforms::Compose compose({
    //     xt::transforms::image::Resize({32, 32}),
    //     xt::transforms::general::Normalize({0.5}, {0.5})
    // });


    std::vector<std::shared_ptr<xt::Module>> transform_list;
    transform_list.push_back(std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{32, 32}));
    transform_list.push_back(std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5}, std::vector<float>{0.5}));

    auto compose= std::make_unique<xt::transforms::Compose>(transform_list);

    auto dataset = xt::datasets::MNIST(
        "/home/kami/Documents/datasets/", xt::datasets::DataMode::TRAIN, false, std::move(compose));






    int num_epochs = 2;
    auto datum = dataset.get(0);
    cout << datum.data.sizes() << endl;

    xt::dataloaders::MyCustomDataLoaderV2 data_loader(dataset, 64, true, 2, /*prefetch_factor=*/2);



    for (int epoch = 1; epoch <= num_epochs; ++epoch)
    {
        std::cout << "\nEpoch: " << epoch << std::endl;
        int batch_count = 0;
        auto epoch_start_time = std::chrono::high_resolution_clock::now();

        for (const auto& batch : data_loader)
        {
            // data_loader.begin() calls reset_epoch()
            torch::Tensor features = batch.first;
            torch::Tensor labels = batch.second;

            // Simulate some training work on the batch
            // std::this_thread::sleep_for(std::chrono::milliseconds(20)); // Uncomment to see prefetching benefit

            std::cout << "  Batch " << ++batch_count << " received. Features: " << features.sizes()
                << ", Labels: " << labels.sizes();
            if (labels.numel() > 0)
            {
                std::cout << " First label: " << labels[0].item<long>();
            }
            std::cout << std::endl;
        }
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - epoch_start_time);
        std::cout << "Epoch " << epoch << " completed in " << duration.count() << "ms. Total batches: " << batch_count
            << std::endl;
        if (batch_count == 0 && dataset.size().value_or(0) > 0)
        {
            std::cerr << "Error: No batches processed for a non-empty dataset in epoch " << epoch << std::endl;
        }
    }


    return 0;
}
