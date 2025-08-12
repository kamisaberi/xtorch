#include <iostream>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <algorithm> // For std::shuffle, std::iota, std::min, std::max
#include <future>    // Not directly used in V2, but good to keep in mind for other patterns
#include <optional>   // For torch::optional and std::optional
#include <chrono>     // For timing in main
#include <xtorch/xtorch.h>
#include  <chrono>

using namespace std;


int main()
{
    int threads = 16;
    std::cout.precision(10);
    torch::set_num_threads(16);  // Use all 16 cores
    std::cout << "Using " << torch::get_num_threads() << " threads for LibTorch" << std::endl;
    int epochs = 10;

    std::vector<std::shared_ptr<xt::Module>> transform_list;
    transform_list.push_back(std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{32, 32}));
    transform_list.push_back(
        std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5}, std::vector<float>{0.5}));
    auto compose = std::make_unique<xt::transforms::Compose>(transform_list);
    auto dataset = xt::datasets::MNIST("/home/kami/Documents/datasets/", xt::datasets::DataMode::TRAIN, false,
                                       std::move(compose));
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 64, true, 16, 2);
    xt::models::LeNet5 model(10);
    model.to(torch::Device(torch::kCPU));
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    auto start_time = std::chrono::steady_clock::now();
    for (int i = 1; i <= epochs; i++)
    {
        int btc = 0;
        for (auto& batch_data : data_loader)
        {
            btc++;
            auto epoch_start = std::chrono::steady_clock::now();

            torch::Tensor data = batch_data.first;
            torch::Tensor target = batch_data.second;

            // if (btc % 20 == 0)
            // {
            //     auto t = std::chrono::steady_clock::now();
            //     auto d = t - epoch_start;
            //     auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(d);
            //     cout << "COPY: " << btc << " DIFF:" << d.count() << endl;
            // }

            auto output_any = model.forward({data});
            // if (btc % 20 == 0)
            // {
            //     auto t = std::chrono::steady_clock::now();
            //     auto d = t - epoch_start;
            //     auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(d);
            //     cout << "FORWARD: " << btc << " DIFF:" << d.count() << endl;
            // }


            auto output = std::any_cast<torch::Tensor>(output_any);
            torch::Tensor loss = torch::nll_loss(output, target);
            // if (btc % 20 == 0)
            // {
            //     auto t = std::chrono::steady_clock::now();
            //     auto d = t - epoch_start;
            //     auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(d);
            //     cout << "LOSS: " << btc << " DIFF:" << d.count() << endl;
            // }

            loss.backward();
            // if (btc % 20 == 0)
            // {
            //     auto t = std::chrono::steady_clock::now();
            //     auto d = t - epoch_start;
            //     auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(d);
            //     cout << "BACKWARD: " << btc << " DIFF:" << d.count() << endl;
            // }


            optimizer.zero_grad();
            // if (btc % 20 == 0)
            // {
            //     auto t = std::chrono::steady_clock::now();
            //     auto d = t - epoch_start;
            //     auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(d);
            //     cout << "ZERO GRAD: " << btc << " DIFF:" << d.count() << endl;
            // }

            optimizer.step();
            // if (btc % 20 == 0)
            // {
            //     auto t = std::chrono::steady_clock::now();
            //     auto d = t - epoch_start;
            //     auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(d);
            //     cout << "STEP: " << btc << " DIFF:" << d.count() << endl;
            // }

            if (btc % 20 == 0)
            {
                cout << "Batch: " << btc << " Loss:" << loss.item() << endl;
            }
        }
    }
    auto end_time = std::chrono::steady_clock::now();
    auto duration = end_time - start_time;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    std::cout << "Total loop duration: " << duration_ms.count() << " milliseconds." << std::endl;

    // auto logger = std::make_shared<xt::LoggingCallback>("[MyTrain]", /*log_every_N_batches=*/20, /*log_time=*/true);
    // xt::Trainer trainer;
    //
    // trainer.set_max_epochs(1).set_optimizer(optimizer)
    //        .set_loss_fn([](const auto& output, const auto& target)
    //        {
    //            return torch::nll_loss(output, target);
    //        })
    //        .add_callback(logger);
    // auto start_time = std::chrono::steady_clock::now();
    // trainer.fit(model, data_loader, &data_loader, torch::Device(torch::kCPU));
    //
    // auto end_time = std::chrono::steady_clock::now();
    // auto duration = end_time - start_time;
    // auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    // std::cout << "Total loop duration: " << duration_ms.count() << " milliseconds." << std::endl;

    return 0;
}
