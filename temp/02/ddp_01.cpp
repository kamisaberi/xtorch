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
struct SimpleNet : torch::nn::Module {
    SimpleNet() {
        fc1 = register_module("fc1", torch::nn::Linear(10, 50));
        fc2 = register_module("fc2", torch::nn::Linear(50, 2));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

// DistributedDataParallel class
class DistributedDataParallel {
public:
    DistributedDataParallel(int rank, int world_size, const std::string& master_ip, int master_port)
        : rank_(rank), world_size_(world_size), master_ip_(master_ip), master_port_(master_port) {
        model_ = std::make_shared<SimpleNet>();
        optimizer_ = std::make_unique<torch::optim::SGD>(model_->parameters(), torch::optim::SGDOptions(0.01));
        setup_communication();
    }

    ~DistributedDataParallel() {
        if (rank_ == 0) {
            close(client_fd_);
            close(server_fd_);
        } else {
            close(sock_fd_);
        }
    }

    void train(const std::vector<std::pair<torch::Tensor, torch::Tensor>>& dataset, int epochs) {
        model_->train();
        auto criterion = torch::nn::CrossEntropyLoss();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0;
            for (const auto& [data, target] : dataset) {
                optimizer_->zero_grad();
                auto output = model_->forward(data);
                auto loss = criterion(output, target);
                loss.backward();

                // Aggregate gradients
                aggregate_gradients();

                // Update model parameters
                optimizer_->step();

                // Synchronize model parameters
                synchronize_model();

                total_loss += loss.item<float>();
            }
            std::cout << "Rank " << rank_ << ", Epoch " << epoch + 1 << ", Loss: "
                      << total_loss / dataset.size() << std::endl;
        }
    }

private:
    void setup_communication() {
        if (rank_ == 0) {
            // Master: Set up server
            server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
            if (server_fd_ < 0) throw std::runtime_error("Socket creation failed");

            struct sockaddr_in server_addr;
            server_addr.sin_family = AF_INET;
            server_addr.sin_addr.s_addr = INADDR_ANY;
            server_addr.sin_port = htons(master_port_);

            if (bind(server_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
                throw std::runtime_error("Bind failed");
            }
            if (listen(server_fd_, world_size_ - 1) < 0) {
                throw std::runtime_error("Listen failed");
            }

            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            client_fd_ = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd_ < 0) throw std::runtime_error("Accept failed");
        } else {
            // Worker: Connect to master
            sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
            if (sock_fd_ < 0) throw std::runtime_error("Socket creation failed");

            struct sockaddr_in server_addr;
            server_addr.sin_family = AF_INET;
            server_addr.sin_port = htons(master_port_);
            if (inet_pton(AF_INET, master_ip_.c_str(), &server_addr.sin_addr) <= 0) {
                throw std::runtime_error("Invalid address");
            }

            while (connect(sock_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
                std::cerr << "Connection failed, retrying..." << std::endl;
                sleep(1);
            }
        }
    }

    void send_data(int fd, const torch::Tensor& tensor) {
        auto data = tensor.data_ptr<float>();
        size_t size = tensor.numel() * sizeof(float);
        uint32_t length = static_cast<uint32_t>(size);
        send(fd, &length, sizeof(length), 0);
        send(fd, data, size, 0);
    }

    torch::Tensor receive_data(int fd) {
        uint32_t length;
        recv(fd, &length, sizeof(length), MSG_WAITALL);
        std::vector<float> buffer(length / sizeof(float));
        recv(fd, buffer.data(), length, MSG_WAITALL);
        return torch::from_blob(buffer.data(), {static_cast<long>(buffer.size())});
    }

    void aggregate_gradients() {
        std::vector<torch::Tensor> all_grads;
        for (auto& param : model_->parameters()) {
            all_grads.push_back(param.grad().clone());
        }

        if (rank_ == 0) {
            // Master: Collect gradients
            std::vector<std::vector<torch::Tensor>> gathered_grads(world_size_);
            gathered_grads[0] = all_grads;

            for (int i = 1; i < world_size_; ++i) {
                std::vector<torch::Tensor> worker_grads;
                for (size_t j = 0; j < all_grads.size(); ++j) {
                    worker_grads.push_back(receive_data(client_fd_));
                }
                gathered_grads[i] = worker_grads;
            }

            // Average gradients
            for (size_t j = 0; j < all_grads.size(); ++j) {
                auto avg_grad = gathered_grads[0][j].clone().zero_();
                for (int i = 0; i < world_size_; ++i) {
                    avg_grad += gathered_grads[i][j];
                }
                avg_grad /= world_size_;
                all_grads[j].copy_(avg_grad);
            }

            // Broadcast averaged gradients
            for (size_t j = 0; j < all_grads.size(); ++j) {
                send_data(client_fd_, all_grads[j]);
            }
        } else {
            // Worker: Send gradients and receive averaged gradients
            for (const auto& grad : all_grads) {
                send_data(sock_fd_, grad);
            }
            for (size_t j = 0; j < all_grads.size(); ++j) {
                all_grads[j] = receive_data(sock_fd_);
            }
        }

        // Update gradients in model
        size_t idx = 0;
        for (auto& param : model_->parameters()) {
            param.grad() = all_grads[idx++].clone();
        }
    }

    void synchronize_model() {
        if (rank_ == 0) {
            // Master: Send model parameters
            for (const auto& param : model_->parameters()) {
                send_data(client_fd_, param);
            }
        } else {
            // Worker: Receive and update model parameters
            for (auto& param : model_->parameters()) {
                auto new_param = receive_data(sock_fd_);
                param.copy_(new_param.reshape(param.sizes()));
            }
        }
    }

    int rank_, world_size_;
    std::string master_ip_;
    int master_port_;
    std::shared_ptr<SimpleNet> model_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    int server_fd_ = -1, client_fd_ = -1, sock_fd_ = -1;
};

// Main function
int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <rank> <master_ip> <master_port>" << std::endl;
        return 1;
    }

    int rank = std::stoi(argv[1]);
    std::string master_ip = argv[2];
    int master_port = std::stoi(argv[3]);
    int world_size = 2; // Two nodes
    int epochs = 5;

    // Create synthetic dataset
    std::vector<std::pair<torch::Tensor, torch::Tensor>> dataset;
    int data_per_node = 50;
    torch::manual_seed(rank); // Different data per node
    for (int i = 0; i < data_per_node; ++i) {
        auto data = torch::randn({1, 10});
        auto target = torch::randint(0, 2, {1});
        dataset.emplace_back(data, target);
    }

    // Initialize DDP
    DistributedDataParallel ddp(rank, world_size, master_ip, master_port);

    // Train
    ddp.train(dataset, epochs);

    return 0;
}