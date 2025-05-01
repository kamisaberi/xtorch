#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <cstdlib>
#include <iostream>

// Dummy model
struct Net : torch::nn::Module {
    torch::nn::Linear fc;
    Net() : fc(10, 10) {
        register_module("fc", fc);
    }
    torch::Tensor forward(torch::Tensor x) {
        return fc->forward(x);
    }
};

int main(int argc, char* argv[]) {
    // === Environment variables ===
    int rank = std::stoi(std::getenv("RANK"));
    int world_size = std::stoi(std::getenv("WORLD_SIZE"));
    std::string master_addr = std::getenv("MASTER_ADDR");
    int master_port = std::stoi(std::getenv("MASTER_PORT"));
    bool is_master = (rank == 0);

    // === Device selection ===
    torch::Device device = torch::kCPU; // Change to kCUDA if using GPUs

    // === Create store and process group ===
    auto store = std::make_shared<torch::distributed::c10d::TCPStore>(
        master_addr, master_port, is_master, world_size);
    auto options = std::make_shared<torch::distributed::c10d::ProcessGroupGloo::Options>();
    options->devices.push_back(torch::distributed::c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));
    auto pg = std::make_shared<torch::distributed::c10d::ProcessGroupGloo>(store, rank, world_size, options);

    // === Model and optimizer ===
    auto model = std::make_shared<Net>();
    model->to(device);
    torch::optim::SGD optimizer(model->parameters(), 0.01);

    // === Broadcast parameters from rank 0 ===
    for (auto& param : model->parameters()) {
        std::vector<at::Tensor> tensors = {param.data()};
        pg->broadcast(tensors)->wait();
    }

    // === Dummy input and target ===
    torch::Tensor input = torch::randn({16, 10}).to(device);
    torch::Tensor target = torch::randn({16, 10}).to(device);

    // === Training loop ===
    for (int epoch = 0; epoch < 5; ++epoch) {
        model->train();

        auto output = model->forward(input);
        auto loss = torch::mse_loss(output, target);

        optimizer.zero_grad();
        loss.backward();

        // All-reduce gradients
        for (auto& param : model->parameters()) {
            if (!param.grad().defined()) continue;
            std::vector<at::Tensor> grads = {param.grad()};
            torch::distributed::c10d::AllreduceOptions opts;
            opts.reduceOp = torch::distributed::c10d::ReduceOp::SUM;
            pg->allreduce(grads, opts)->wait();
            param.grad().div_(world_size);
        }

        optimizer.step();

        if (rank == 0) {
            std::cout << "[Epoch " << epoch << "] Loss: " << loss.item<float>() << std::endl;
        }
    }

    pg->barrier()->wait(); // Ensure all processes finish together
    return 0;
}
