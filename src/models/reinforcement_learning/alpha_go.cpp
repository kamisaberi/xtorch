#include "include/models/reinforcement_learning/alpha_go.h"


using namespace std;

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <queue>
#include <map>
#include <filesystem>
#include <fstream>
#include <sstream>

// Game State for 5x5 Go (simplified)
class GoGame {
public:
    GoGame(int size = 5) : size_(size), board_(size * size, 0), current_player_(1) {
        // 0: empty, 1: black, -1: white
    }

    bool is_valid_move(int x, int y) const {
        if (x < 0 || x >= size_ || y < 0 || y >= size_) return false;
        return board_[x * size_ + y] == 0;
    }

    bool make_move(int x, int y) {
        if (!is_valid_move(x, y)) return false;
        board_[x * size_ + y] = current_player_;
        current_player_ = -current_player_; // Switch player
        return true;
    }

    int get_winner() const {
        // Simplified: count stones (real Go uses territory + captures)
        int black_score = 0, white_score = 0;
        for (int v : board_) {
            if (v == 1) black_score++;
            else if (v == -1) white_score++;
        }
        if (black_score > white_score) return 1;
        if (white_score > black_score) return -1;
        return 0;
    }

    bool is_game_over() const {
        // Game ends when board is full or both players pass (simplified: only full board)
        for (int v : board_) {
            if (v == 0) return false;
        }
        return true;
    }

    torch::Tensor get_state() const {
        std::vector<float> state;
        for (int v : board_) {
            state.push_back(static_cast<float>(v)); // Black: 1, White: -1, Empty: 0
        }
        return torch::tensor(state).view({1, 1, size_, size_}); // [batch=1, channels=1, h, w]
    }

    std::vector<std::pair<int, int>> get_legal_moves() const {
        std::vector<std::pair<int, int>> moves;
        for (int x = 0; x < size_; ++x) {
            for (int y = 0; y < size_; ++y) {
                if (is_valid_move(x, y)) moves.emplace_back(x, y);
            }
        }
        return moves;
    }

    int current_player() const { return current_player_; }
    int size() const { return size_; }

private:
    int size_;
    std::vector<int> board_;
    int current_player_;
};

// Policy Network
struct PolicyNetworkImpl : torch::nn::Module {
    PolicyNetworkImpl(int board_size) {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, 64, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
        fc = register_module("fc", torch::nn::Linear(128 * board_size * board_size, board_size * board_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x)); // [batch, 64, h, w]
        x = torch::relu(conv2->forward(x)); // [batch, 128, h, w]
        x = x.view({x.size(0), -1}); // [batch, 128*h*w]
        x = torch::softmax(fc->forward(x), 1); // [batch, h*w]
        return x;
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc{nullptr};
};
TORCH_MODULE(PolicyNetwork);

// Value Network
struct ValueNetworkImpl : torch::nn::Module {
    ValueNetworkImpl(int board_size) {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, 64, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
        fc1 = register_module("fc1", torch::nn::Linear(128 * board_size * board_size, 256));
        fc2 = register_module("fc2", torch::nn::Linear(256, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x)); // [batch, 64, h, w]
        x = torch::relu(conv2->forward(x)); // [batch, 128, h, w]
        x = x.view({x.size(0), -1}); // [batch, 128*h*w]
        x = torch::relu(fc1->forward(x)); // [batch, 256]
        x = torch::tanh(fc2->forward(x)); // [batch, 1], win prob in [-1, 1]
        return x;
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};
TORCH_MODULE(ValueNetwork);

// MCTS Node
struct MCTSNode {
    GoGame state;
    MCTSNode* parent;
    std::vector<MCTSNode*> children;
    int action_idx; // Move that led to this state
    int visits;
    double total_value;
    double prior_prob;

    MCTSNode(const GoGame& s, MCTSNode* p = nullptr, int action = -1, double prior = 0.0)
        : state(s), parent(p), action_idx(action), visits(0), total_value(0.0), prior_prob(prior) {}
    ~MCTSNode() { for (auto c : children) delete c; }

    double ucb_score(double c_puct = 1.0) const {
        if (visits == 0) return std::numeric_limits<double>::infinity();
        double q = total_value / visits;
        double u = c_puct * prior_prob * std::sqrt(parent->visits) / (1 + visits);
        return q + u;
    }
};

// Monte Carlo Tree Search
class MCTS {
public:
    MCTS(PolicyNetwork policy_net, ValueNetwork value_net, int board_size, int num_simulations = 100)
        : policy_net_(policy_net), value_net_(value_net), board_size_(board_size), num_simulations_(num_simulations) {
        policy_net_->eval();
        value_net_->eval();
        device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        policy_net_->to(device_);
        value_net_->to(device_);
    }

    std::vector<float> search(GoGame& game) {
        root_ = new MCTSNode(game);
        for (int i = 0; i < num_simulations_; ++i) {
            simulate(root_);
        }
        std::vector<float> probs(board_size_ * board_size_, 0.0);
        for (auto child : root_->children) {
            probs[child->action_idx] = static_cast<float>(child->visits) / root_->visits;
        }
        delete root_;
        return probs;
    }

private:
    void simulate(MCTSNode* node) {
        MCTSNode* leaf = select(node);
        double value = evaluate(leaf);
        backup(leaf, value);
    }

    MCTSNode* select(MCTSNode* node) {
        while (!node->children.empty()) {
            auto it = std::max_element(node->children.begin(), node->children.end(),
                [](MCTSNode* a, MCTSNode* b) { return a->ucb_score() < b->ucb_score(); });
            node = *it;
        }
        if (!node->state.is_game_over() && node->visits > 0) {
            expand(node);
            if (!node->children.empty()) node = node->children[0];
        }
        return node;
    }

    void expand(MCTSNode* node) {
        auto state_tensor = node->state.get_state().to(device_);
        torch::NoGradGuard no_grad;
        auto policy_probs = policy_net_->forward(state_tensor).cpu().squeeze(0).accessor<float, 1>();
        auto legal_moves = node->state.get_legal_moves();
        for (const auto& move : legal_moves) {
            GoGame next_state = node->state;
            if (next_state.make_move(move.first, move.second)) {
                int idx = move.first * board_size_ + move.second;
                auto child = new MCTSNode(next_state, node, idx, policy_probs[idx]);
                node->children.push_back(child);
            }
        }
    }

    double evaluate(MCTSNode* node) {
        if (node->state.is_game_over()) {
            return node->state.get_winner() * node->state.current_player();
        }
        auto state_tensor = node->state.get_state().to(device_);
        torch::NoGradGuard no_grad;
        auto value = value_net_->forward(state_tensor).cpu().item<float>();
        return static_cast<double>(value);
    }

    void backup(MCTSNode* node, double value) {
        while (node) {
            node->visits++;
            node->total_value += value;
            value = -value; // Flip for opponent
            node = node->parent;
        }
    }

    PolicyNetwork policy_net_;
    ValueNetwork value_net_;
    int board_size_, num_simulations_;
    MCTSNode* root_;
    torch::Device device_;
};

// Self-Play Dataset
class SelfPlayDataset : public torch::data::Dataset<SelfPlayDataset> {
public:
    SelfPlayDataset(const std::string& data_dir, int board_size)
        : board_size_(board_size) {
        for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
            if (entry.path().extension() == ".txt") {
                data_files_.push_back(entry.path().string());
            }
        }
    }

    torch::data::Example<> get(size_t index) override {
        std::ifstream file(data_files_[index]);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + data_files_[index]);
        }
        std::vector<float> state(board_size_ * board_size_);
        std::vector<float> policy(board_size_ * board_size_);
        float value;
        std::string line;
        // Read state
        std::getline(file, line);
        std::istringstream iss_state(line);
        for (int i = 0; i < board_size_ * board_size_; ++i) {
            iss_state >> state[i];
        }
        // Read policy
        std::getline(file, line);
        std::istringstream iss_policy(line);
        for (int i = 0; i < board_size_ * board_size_; ++i) {
            iss_policy >> policy[i];
        }
        // Read value
        std::getline(file, line);
        std::istringstream iss_value(line);
        iss_value >> value;

        auto state_tensor = torch::tensor(state).view({1, 1, board_size_, board_size_});
        auto policy_tensor = torch::tensor(policy);
        auto value_tensor = torch::tensor(value);
        return {torch::cat({state_tensor.unsqueeze(0), policy_tensor.unsqueeze(0), value_tensor.unsqueeze(0)}, 0), torch::Tensor()};
    }

    torch::optional<size_t> size() const override {
        return data_files_.size();
    }

private:
    std::vector<std::string> data_files_;
    int board_size_;
};

// Generate Self-Play Data
void generate_self_play_data(PolicyNetwork policy_net, ValueNetwork value_net, int board_size, int num_games, const std::string& output_dir) {
    MCTS mcts(policy_net, value_net, board_size, 100);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    std::filesystem::create_directory(output_dir);
    for (int game_idx = 0; game_idx < num_games; ++game_idx) {
        GoGame game(board_size);
        std::vector<std::tuple<torch::Tensor, std::vector<float>, float>> game_data;
        while (!game.is_game_over()) {
            auto probs = mcts.search(game);
            auto state = game.get_state().squeeze(0); // [1, h, w]
            auto legal_moves = game.get_legal_moves();
            float temperature = 1.0;
            std::vector<float> tempered_probs(probs.size(), 0.0);
            float sum = 0.0;
            for (const auto& move : legal_moves) {
                int idx = move.first * board_size + move.second;
                tempered_probs[idx] = std::pow(probs[idx], 1.0 / temperature);
                sum += tempered_probs[idx];
            }
            for (auto& p : tempered_probs) p /= sum;

            // Sample move
            float r = dist(rng);
            float cumsum = 0.0;
            int move_idx = 0;
            for (size_t i = 0; i < tempered_probs.size(); ++i) {
                cumsum += tempered_probs[i];
                if (r <= cumsum) {
                    move_idx = i;
                    break;
                }
            }
            int x = move_idx / board_size;
            int y = move_idx % board_size;
            game_data.emplace_back(state, tempered_probs, 0.0); // Value placeholder
            game.make_move(x, y);
        }
        // Assign final value
        float winner = static_cast<float>(game.get_winner());
        for (auto& [s, p, v] : game_data) {
            v = winner;
            winner = -winner; // Flip for opponent
        }
        // Save game data
        std::ofstream out(output_dir + "/game_" + std::to_string(game_idx) + ".txt");
        for (const auto& [state, probs, value] : game_data) {
            for (int i = 0; i < board_size * board_size; ++i) {
                out << state[0][i].item<float>() << " ";
            }
            out << "\n";
            for (float p : probs) {
                out << p << " ";
            }
            out << "\n" << value << "\n";
        }
        out.close();
        std::cout << "Generated game " << game_idx + 1 << "/" << num_games << "\r" << std::flush;
    }
    std::cout << "\n";
}

int main() {
    try {
        // Set device
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

        // Hyperparameters
        const int board_size = 5;
        const int batch_size = 32;
        const int supervised_epochs = 5;
        const int rl_epochs = 5;
        const float learning_rate = 0.001;
        const int num_self_play_games = 100;

        // Initialize models
        PolicyNetwork policy_net(board_size);
        ValueNetwork value_net(board_size);
        policy_net->to(device);
        value_net->to(device);

        // Optimizers
        torch::optim::Adam policy_optimizer(policy_net->parameters(), torch::optim::AdamOptions(learning_rate));
        torch::optim::Adam value_optimizer(value_net->parameters(), torch::optim::AdamOptions(learning_rate));

        // Supervised learning phase (simulated with toy data)
        std::cout << "Supervised learning phase...\n";
        policy_net->train();
        value_net->train();
        for (int epoch = 0; epoch < supervised_epochs; ++epoch) {
            // Simulate expert data (in practice, use human games)
            SelfPlayDataset dummy_dataset("./data/supervised", board_size); // Assume pre-generated
            auto data_loader = torch::data::make_data_loader(
                dummy_dataset.map(torch::data::transforms::Stack<>()),
                torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

            float policy_loss_sum = 0.0, value_loss_sum = 0.0;
            int batch_count = 0;
            for (auto& batch : *data_loader) {
                auto data = batch.data.to(device);
                auto states = data[0].to(torch::kFloat32); // [batch, 1, h, w]
                auto target_probs = data[1].to(torch::kFloat32); // [batch, h*w]
                auto target_values = data[2].to(torch::kFloat32); // [batch]

                policy_optimizer.zero_grad();
                value_optimizer.zero_grad();

                auto policy_probs = policy_net->forward(states); // [batch, h*w]
                auto values = value_net->forward(states).squeeze(1); // [batch]

                auto policy_loss = torch::nn::functional::cross_entropy(policy_probs, target_probs);
                auto value_loss = torch::nn::functional::mse_loss(values, target_values);

                policy_loss.backward();
                value_loss.backward();

                policy_optimizer.step();
                value_optimizer.step();

                policy_loss_sum += policy_loss.item<float>();
                value_loss_sum += value_loss.item<float>();
                batch_count++;
            }
            std::cout << "Supervised Epoch [" << epoch + 1 << "/" << supervised_epochs
                      << "] Policy Loss: " << policy_loss_sum / batch_count
                      << ", Value Loss: " << value_loss_sum / batch_count << std::endl;
        }
        torch::save(policy_net, "policy_supervised.pt");
        torch::save(value_net, "value_supervised.pt");

        // Reinforcement learning phase
        std::cout << "Reinforcement learning phase...\n";
        for (int epoch = 0; epoch < rl_epochs; ++epoch) {
            // Generate self-play data
            generate_self_play_data(policy_net, value_net, board_size, num_self_play_games, "./data/self_play");

            // Train on self-play data
            SelfPlayDataset dataset("./data/self_play", board_size);
            auto data_loader = torch::data::make_data_loader(
                dataset.map(torch::data::transforms::Stack<>()),
                torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

            float policy_loss_sum = 0.0, value_loss_sum = 0.0;
            int batch_count = 0;
            for (auto& batch : *data_loader) {
                auto data = batch.data.to(device);
                auto states = data[0].to(torch::kFloat32); // [batch, 1, h, w]
                auto target_probs = data[1].to(torch::kFloat32); // [batch, h*w]
                auto target_values = data[2].to(torch::kFloat32); // [batch]

                policy_optimizer.zero_grad();
                value_optimizer.zero_grad();

                auto policy_probs = policy_net->forward(states); // [batch, h*w]
                auto values = value_net->forward(states).squeeze(1); // [batch]

                auto policy_loss = torch::nn::functional::cross_entropy(policy_probs, target_probs);
                auto value_loss = torch::nn::functional::mse_loss(values, target_values);

                policy_loss.backward();
                value_loss.backward();

                policy_optimizer.step();
                value_optimizer.step();

                policy_loss_sum += policy_loss.item<float>();
                value_loss_sum += value_loss.item<float>();
                batch_count++;
            }
            std::cout << "RL Epoch [" << epoch + 1 << "/" << rl_epochs
                      << "] Policy Loss: " << policy_loss_sum / batch_count
                      << ", Value Loss: " << value_loss_sum / batch_count << std::endl;

            // Save checkpoint
            torch::save(policy_net, "policy_rl_epoch_" + std::to_string(epoch + 1) + ".pt");
            torch::save(value_net, "value_rl_epoch_" + std::to_string(epoch + 1) + ".pt");
        }

        // Save final models
        torch::save(policy_net, "policy_final.pt");
        torch::save(value_net, "value_final.pt");
        std::cout << "Saved final models: policy_final.pt, value_final.pt\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

namespace xt::models
{
    AlphaGo::AlphaGo(int num_classes, int in_channels)
    {
    }

    AlphaGo::AlphaGo(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void AlphaGo::reset()
    {
    }

    auto AlphaGo::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];

        return x;
    }
}
