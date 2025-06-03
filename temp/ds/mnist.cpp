#include <iostream>
#include <torch/torch.h>
#include <torch/data/datasets.h> // For torch::data::Dataset
#include <torch/data/example.h>  // For torch::data::Example
#include <vector>
#include <string>
#include <algorithm> // For std::shuffle, std::iota, std::min, std::max
#include <future>    // Not directly used in V2, but good to keep in mind for other patterns
#include <optional>   // For torch::optional and std::optional
#include <chrono>     // For timing in main
#include "include/data_loaders/extended_data_loader.h"

#include "include/datasets/computer_vision/image_classification/mnist.h"

#include "include/models/computer_vision/image_classification/lenet5.h"
#include "include/trainers/logging_callback.h"
#include "include/trainers/trainer.h"

using namespace std;

#include <torch/nn/parallel/data_parallel.h>


int main()
{
    std::cout.precision(10);

    std::vector<std::shared_ptr<xt::Module>> transform_list;
    transform_list.push_back(std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{32, 32}));
    transform_list.push_back(
        std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5}, std::vector<float>{0.5}));

    auto compose = std::make_unique<xt::transforms::Compose>(transform_list);

    auto dataset = xt::datasets::MNIST(
        "/home/kami/Documents/datasets/", xt::datasets::DataMode::TRAIN, false, std::move(compose));


    int num_epochs = 2;
    auto datum = dataset.get(0);
    cout << datum.data.sizes() << endl;

    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 64, true, 2, /*prefetch_factor=*/2);

    xt::models::LeNet5 model(10);
    model.to(torch::Device(torch::kCPU));
    model.train();

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    auto logger = std::make_shared<xt::LoggingCallback>("[MyTrain]", /*log_every_N_batches=*/20, /*log_time=*/true);
    xt::Trainer trainer;
    trainer.set_max_epochs(10)
           .set_optimizer(optimizer)
           .set_loss_fn([](auto output, auto target)
           {
               return torch::nll_loss(output, target);
           })
           // .set_lr_scheduler(scheduler) // Optional
           .add_callback(logger) // Add the logger
        // .enable_checkpointing(...) // Optional
        ;

    trainer.fit(model, data_loader, &data_loader, torch::Device(torch::kCPU));


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
