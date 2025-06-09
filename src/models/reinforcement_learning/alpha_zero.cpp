#include "include/models/reinforcement_learning/alpha_zero.h"


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
        current_player_ = -current_player_;
        return true;
    }

    int get_winner() const {
        // Simplified: count stones
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
        for (int v : board_) {
            if (v == 0) return false;
        }
        return true;
    }

    torch::Tensor get_state() const {
        std::vector<float> state;
        for (int v : board_) {
            state.push_back(static_cast<float>(v));
        }
        // Add player plane
        std::vector<float> player_plane(size_ * size_, current_player_);
        state.insert(state.end(), player_plane.begin(), player_plane.end());
        return torch::tensor(state).view({1, 2, size_, size_}); // [batch=1, channels=2, h, w]
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

// Residual Block
struct ResidualBlockImpl : torch::nn::Module {
    ResidualBlockImpl(int channels) {
        conv1 = register_module("conv1", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(channels));
        conv2 = register_module("conv2", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(channels));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto residual = x;
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = bn2->forward(conv2->forward(x));
        x = x + residual;
        return torch::relu(x);
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
};
TORCH_MODULE(ResidualBlock);

// AlphaZero Network (Policy + Value)
struct AlphaZeroNetworkImpl : torch::nn::Module {
    AlphaZeroNetworkImpl(int board_size, int num_blocks = 4, int channels = 64) : board_size_(board_size) {
        conv_in = register_module("conv_in", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(2, channels, 3).padding(1)));
        bn_in = register_module("bn_in", torch::nn::BatchNorm2d(channels));
        for (int i = 0; i < num_blocks; ++i) {
            res_blocks->push_back(ResidualBlock(channels));
            register_module("res_block_" + std::to_string(i), res_blocks->back());
        }
        // Policy head
        policy_conv = register_module("policy_conv", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(channels, 2, 1)));
        policy_bn = register_module("policy_bn", torch::nn::BatchNorm2d(2));
        policy_fc = register_module("policy_fc", torch::nn::Linear(2 * board_size * board_size, board_size * board_size));
        // Value head
        value_conv = register_module("value_conv", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(channels, 1, 1)));
        value_bn = register_module("value_bn", torch::nn::BatchNorm2d(1));
        value_fc1 = register_module("value_fc1", torch::nn::Linear(board_size * board_size, 256));
        value_fc2 = register_module("value_fc2", torch::nn::Linear(256, 1));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(bn_in->forward(conv_in->forward(x))); // [batch, channels, h, w]
        for (auto& block : *res_blocks) {
            x = block->forward(x);
        }
        // Policy head
        auto policy = torch::relu(policy_bn->forward(policy_conv->forward(x))); // [batch, 2, h, w]
        policy = policy.view({policy.size(0), -1}); // [batch, 2*h*w]
        policy = torch::softmax(policy_fc->forward(policy), 1); // [batch, h*w]
        // Value head
        auto value = torch::relu(value_bn->forward(value_conv->forward(x))); // [batch, 1, h, w]
        value = value.view({value.size(0), -1}); // [batch, h*w]
        value = torch::relu(value_fc1->forward(value)); // [batch, 256]
        value = torch::tanh(value_fc2->forward(value)); // [batch, 1]
        return {policy, value};
    }

    int board_size_;
    torch::nn::Conv2d conv_in{nullptr}, policy_conv{nullptr}, value_conv{nullptr};
    torch::nn::BatchNorm2d bn_in{nullptr}, policy_bn{nullptr}, value_bn{nullptr};
    torch::nn::Linear policy_fc{nullptr}, value_fc1{nullptr}, value_fc2{nullptr};
    torch::nn::ModuleList res_blocks{torch::nn::ModuleList()};
};
TORCH_MODULE(AlphaZeroNetwork);

// MCTS Node
struct MCTSNode {
    GoGame state;
    MCTSNode* parent;
    std::vector<MCTSNode*> children;
    int action_idx;
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
    MCTS(AlphaZeroNetwork net, int board_size, int num_simulations = 100)
            : net_(net), board_size_(board_size), num_simulations_(num_simulations) {
        net_->eval();
        device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        net_->to(device_);
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
        auto [policy_probs, value] = net_->forward(state_tensor);
        policy_probs = policy_probs.cpu().squeeze(0).accessor<float, 1>();
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
        auto [_, value] = net_->forward(state_tensor);
        return value.cpu().item<float>();
    }

    void backup(MCTSNode* node, double value) {
        while (node) {
            node->visits++;
            node->total_value += value;
            value = -value;
            node = node->parent;
        }
    }

    AlphaZeroNetwork net_;
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
        std::vector<float> state(board_size_ * board_size * 2);
        std::vector<float> policy(board_size_ * board_size);
        float value;
        std::string line;
        // Read state
        std::getline(file, line);
        std::istringstream iss_state(line);
        for (int i = 0; i < board_size_ * board_size * 2; ++i) {
            iss_state >> state[i];
        }
        // Read policy
        std::getline(file, line);
        std::istringstream iss_policy(line);
        for (int i = 0; i < board_size_ * board_size; ++i) {
            iss_policy >> policy[i];
        }
        // Read value
        std::getline(file, line);
        std::istringstream iss_value(line);
        iss_value >> value;

        auto state_tensor = torch::tensor(state).view({1, 2, board_size_, board_size_});
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
void generate_self_play_data(AlphaZeroNetwork net, int board_size, int num_games, const std::string& output_dir) {
MCTS mcts(net, board_size, 100);
std::random_device rd;
std::mt19937 rng(rd());
std::uniform_real_distribution<float> dist(0.0, 1.0);

std::filesystem::create_directory(output_dir);
for (int game_idx = 0; game_idx < num_games; ++game_idx) {
GoGame game(board_size);
std::vector<std::tuple<torch::Tensor, std::vector<float>, float>> game_data;
float temperature = 1.0;
int move_count = 0;
while (!game.is_game_over()) {
if (move_count++ > 10) temperature = 0.1; // Reduce exploration after 10 moves
auto probs = mcts.search(game);
auto state = game.get_state().squeeze(0); // [2, h, w]
auto legal_moves = game.get_legal_moves();
std::vector<float> tempered_probs(probs.size(), 0.0);
float sum = 0.0;
for (const auto& move : legal_moves) {
int idx = move.first * board_size + move.second;
tempered_probs[idx] = std::pow(probs[idx], 1.0 / temperature);
sum += tempered_probs[idx];
}
for (auto& p : tempered_probs) p /= (sum + 1e-10);

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
game_data.emplace_back(state, tempered_probs, 0.0);
game.make_move(x, y);
}
// Assign final value
float winner = static_cast<float>(game.get_winner());
for (auto& [s, p, v] : game_data) {
v = winner;
winner = -winner;
}
// Save game data
std::ofstream out(output_dir + "/game_" + std::to_string(game_idx) + ".txt");
for (const auto& [state, probs, value] : game_data) {
for (int i = 0; i < 2 * board_size * board_size; ++i) {
out << state.view(-1)[i].item<float>() << " ";
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
        const int num_epochs = 10;
        const float learning_rate = 0.001;
        const int num_self_play_games = 100;
        const int num_res_blocks = 4;
        const int num_channels = 64;

        // Initialize model
        AlphaZeroNetwork net(board_size, num_res_blocks, num_channels);
        net->to(device);

        // Optimizer
        torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(learning_rate));

        // Training loop
        net->train();
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // Generate self-play data
            generate_self_play_data(net, board_size, num_self_play_games, "./data/self_play");

            // Load dataset
            SelfPlayDataset dataset("./data/self_play", board_size);
            auto data_loader = torch::data::make_data_loader(
                    dataset.map(torch::data::transforms::Stack<>()),
                    torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

            float policy_loss_sum = 0.0, value_loss_sum = 0.0;
            int batch_count = 0;
            for (auto& batch : *data_loader) {
                auto data = batch.data.to(device);
                auto states = data[0].to(torch::kFloat32); // [batch, 2, h, w]
                auto target_probs = data[1].to(torch::kFloat32); // [batch, h*w]
                auto target_values = data[2].to(torch::kFloat32); // [batch]

                optimizer.zero_grad();
                auto [policy_probs, values] = net->forward(states); // [batch, h*w], [batch, 1]
                values = values.squeeze(1); // [batch]

                auto policy_loss = -torch::sum(target_probs * torch::log(policy_probs + 1e-10), 1).mean();
                auto value_loss = torch::nn::functional::mse_loss(values, target_values);

                auto total_loss = policy_loss + value_loss;
                total_loss.backward();
                optimizer.step();

                policy_loss_sum += policy_loss.item<float>();
                value_loss_sum += value_loss.item<float>();
                batch_count++;
            }
            std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs
                      << "] Policy Loss: " << policy_loss_sum / batch_count
                      << ", Value Loss: " << value_loss_sum / batch_count << std::endl;

            // Save checkpoint
            if ((epoch + 1) % 5 == 0) {
                torch::save(net, "alphazero_epoch_" + std::to_string(epoch + 1) + ".pt");
                std::cout << "Saved checkpoint: alphazero_epoch_" << epoch + 1 << ".pt" << std::endl;
            }
        }

        // Save final model
        torch::save(net, "alphazero_final.pt");
        std::cout << "Saved final model: alphazero_final.pt\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}



namespace xt::models
{
    AlphaZero::AlphaZero(int num_classes, int in_channels)
    {
    }

    AlphaZero::AlphaZero(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void AlphaZero::reset()
    {
    }

    auto AlphaZero::forward(std::initializer_list<std::any> tensors) -> std::any
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
