#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <memory>

// Simple neural network model
struct SimpleNet : torch::nn::Module
{
    SimpleNet()
    {
        fc1 = register_module("fc1", torch::nn::Linear(10, 50));
        fc2 = register_module("fc2", torch::nn::Linear(50, 2));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

// DistributedDataParallel class
class DistributedDataParallel
{
public:
    DistributedDataParallel(int rank, int world_size, const std::string& master_ip, int master_port);

    ~DistributedDataParallel();

    void train(const std::vector<std::pair<torch::Tensor, torch::Tensor>>& dataset, int epochs);

private:
    void setup_communication();
    void send_data(int fd, const torch::Tensor& tensor);

    torch::Tensor receive_data(int fd);

    void aggregate_gradients();
    void synchronize_model();
    int rank_, world_size_;
    std::string master_ip_;
    int master_port_;
    std::shared_ptr<SimpleNet> model_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    std::vector<torch::Tensor> parameters_; // Store parameters for non-const access
    int server_fd_ = -1, client_fd_ = -1, sock_fd_ = -1;
};
