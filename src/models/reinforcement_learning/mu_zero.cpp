#include <models/reinforcement_learning/mu_zero.h>


using namespace std;

// #include <torch/torch.h>
// #include <vector>
// #include <random>
// #include <iostream>
// #include <deque>
// #include <cmath>
// #include <map>
//
// // Simulated CartPole Environment
// struct CartPoleEnv {
//     std::vector<float> state;
//     float gravity = 9.8, masscart = 1.0, masspole = 0.1, length = 0.5;
//     float tau = 0.02, force_mag = 10.0;
//     bool done = false;
//     int steps = 0;
//     const int max_steps = 200;
//     std::mt19937 gen;
//
//     CartPoleEnv() : state(4, 0.0), gen(std::random_device{}()) {
//         reset();
//     }
//
//     void reset() {
//         std::uniform_real_distribution<float> dist(-0.05, 0.05);
//         state = {dist(gen), dist(gen), dist(gen), dist(gen)};
//         done = false;
//         steps = 0;
//     }
//
//     std::tuple<std::vector<float>, float, bool> step(int action) {
//         float x = state[0], x_dot = state[1], theta = state[2], theta_dot = state[3];
//         float force = (action == 1) ? force_mag : -force_mag;
//         float costheta = std::cos(theta), sintheta = std::sin(theta);
//         float temp = (force + masspole * length * theta_dot * theta_dot * sintheta) / (masscart + masspole);
//         float theta_acc = (gravity * sintheta - costheta * temp) /
//                           (length * (4.0 / 3.0 - masspole * costheta * costheta / (masscart + masspole)));
//         float x_acc = temp - masspole * length * theta_acc * costheta / (masscart + masspole);
//
//         x += tau * x_dot;
//         x_dot += tau * x_acc;
//         theta += tau * theta_dot;
//         theta_dot += tau * theta_acc;
//
//         state = {x, x_dot, theta, theta_dot};
//         steps++;
//
//         float reward = 1.0;
//         done = (std::abs(x) > 2.4 || std::abs(theta) > 12.0 * M_PI / 180.0 || steps >= max_steps);
//         if (done && steps < max_steps) {
//             reward = -10.0;
//         }
//
//         return {state, reward, done};
//     }
// };
//
// // Neural Networks for MuZero
// struct RepresentationNetwork : torch::nn::Module {
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
//
//     RepresentationNetwork(int state_dim, int hidden_dim) {
//         fc1 = register_module("fc1", torch::nn::Linear(state_dim, 128));
//         fc2 = register_module("fc2", torch::nn::Linear(128, hidden_dim));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(fc1->forward(x));
//         return fc2->forward(x);
//     }
// };
//
// struct DynamicsNetwork : torch::nn::Module {
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
//
//     DynamicsNetwork(int hidden_dim, int action_dim) {
//         fc1 = register_module("fc1", torch::nn::Linear(hidden_dim + action_dim, 128));
//         fc2 = register_module("fc2", torch::nn::Linear(128, hidden_dim));
//     }
//
//     torch::Tensor forward(torch::Tensor hidden, torch::Tensor action) {
//         auto x = torch::cat({hidden, action}, -1);
//         x = torch::relu(fc1->forward(x));
//         return fc2->forward(x);
//     }
// };
//
// struct PredictionNetwork : torch::nn::Module {
//     torch::nn::Linear fc_policy{nullptr}, fc_value{nullptr};
//
//     PredictionNetwork(int hidden_dim, int action_dim) {
//         fc_policy = register_module("fc_policy", torch::nn::Linear(hidden_dim, action_dim));
//         fc_value = register_module("fc_value", torch::nn::Linear(hidden_dim, 1));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         auto policy_logits = fc_policy->forward(x);
//         auto policy = torch::softmax(policy_logits, -1);
//         auto value = fc_value->forward(x);
//         return {policy, value};
//     }
// };
//
// // Replay Buffer for Trajectories
// struct ReplayBuffer {
//     struct Trajectory {
//         std::vector<std::vector<float>> states;
//         std::vector<int> actions;
//         std::vector<float> rewards;
//         std::vector<std::vector<float>> policy_targets;
//         std::vector<float> value_targets;
//     };
//
//     std::deque<Trajectory> buffer;
//     size_t capacity;
//     std::mt19937 gen;
//
//     ReplayBuffer(size_t capacity_) : capacity(capacity_), gen(std::random_device{}()) {}
//
//     void push(const Trajectory& traj) {
//         if (buffer.size() >= capacity) {
//             buffer.pop_front();
//         }
//         buffer.push_back(traj);
//     }
//
//     Trajectory sample(size_t batch_size) {
//         std::uniform_int_distribution<size_t> dist(0, buffer.size() - 1);
//         auto traj = buffer[dist(gen)];
//         return traj;
//     }
//
//     size_t size() const { return buffer.size(); }
// };
//
// // Simplified MCTS Node
// struct MCTSNode {
//     int action = -1;
//     float prior = 0.0;
//     float value = 0.0;
//     float reward = 0.0;
//     int visit_count = 0;
//     std::vector<float> hidden_state;
//     std::map<int, std::shared_ptr<MCTSNode>> children;
//
//     MCTSNode(int action_, float prior_, const std::vector<float>& hidden_state_)
//         : action(action_), prior(prior_), hidden_state(hidden_state_) {}
// };
//
// // MuZero Agent
// class MuZeroAgent {
// public:
//     RepresentationNetwork rep_net;
//     DynamicsNetwork dyn_net;
//     PredictionNetwork pred_net;
//     ReplayBuffer replay_buffer;
//     torch::optim::Adam optimizer;
//     torch::Device device;
//     float gamma, dirichlet_alpha, exploration_fraction;
//     int state_dim, action_dim, hidden_dim, num_simulations;
//     std::mt19937 gen;
//
//     MuZeroAgent(int state_dim_, int action_dim_, int hidden_dim_, float gamma_, float lr_,
//                 size_t buffer_capacity_, int num_simulations_, float dirichlet_alpha_, float exploration_fraction_)
//         : rep_net(state_dim_, hidden_dim_),
//           dyn_net(hidden_dim_, action_dim_),
//           pred_net(hidden_dim_, action_dim_),
//           replay_buffer(buffer_capacity_),
//           optimizer({rep_net.parameters(), dyn_net.parameters(), pred_net.parameters()}, torch::optim::AdamOptions(lr_)),
//           device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
//           gamma(gamma_),
//           dirichlet_alpha(dirichlet_alpha_),
//           exploration_fraction(exploration_fraction_),
//           state_dim(state_dim_),
//           action_dim(action_dim_),
//           hidden_dim(hidden_dim_),
//           num_simulations(num_simulations_),
//           gen(std::random_device{}()) {
//         rep_net.to(device);
//         dyn_net.to(device);
//         pred_net.to(device);
//     }
//
//     std::vector<float> add_dirichlet_noise(const std::vector<float>& priors) {
//         std::gamma_distribution<float> gamma_dist(dirichlet_alpha, 1.0);
//         std::vector<float> noise(action_dim);
//         float sum = 0.0;
//         for (auto& n : noise) {
//             n = gamma_dist(gen);
//             sum += n;
//         }
//         for (auto& n : noise) n /= sum;
//         std::vector<float> noisy_priors(action_dim);
//         for (int i = 0; i < action_dim; ++i) {
//             noisy_priors[i] = (1 - exploration_fraction) * priors[i] + exploration_fraction * noise[i];
//         }
//         return noisy_priors;
//     }
//
//     float ucb_score(const std::shared_ptr<MCTSNode>& node, int parent_visits, float c1 = 1.25, float c2 = 19652.0) {
//         float prior = node->prior;
//         float value = node->visit_count > 0 ? node->value / node->visit_count : 0.0;
//         float exploration = c1 + std::log((parent_visits + c2 + 1) / c2);
//         float ucb = value + prior * std::sqrt(parent_visits) / (node->visit_count + 1) * exploration;
//         return ucb;
//     }
//
//     void expand_node(std::shared_ptr<MCTSNode> node) {
//         torch::NoGradGuard no_grad;
//         auto hidden_tensor = torch::tensor(node->hidden_state).view({1, hidden_dim}).to(device);
//         auto [policy, value] = pred_net.forward(hidden_tensor);
//         node->value = value.cpu().item<float>();
//         auto policy_probs = policy.cpu().squeeze(0).data_ptr<float>();
//         std::vector<float> priors(action_dim);
//         std::copy(policy_probs, policy_probs + action_dim, priors.begin());
//
//         if (node->visit_count == 0) {
//             priors = add_dirichlet_noise(priors);
//         }
//
//         for (int a = 0; a < action_dim; ++a) {
//             auto action_tensor = torch::zeros({1, action_dim}).to(device);
//             action_tensor[0][a] = 1.0;
//             auto next_hidden = dyn_net.forward(hidden_tensor, action_tensor).cpu().squeeze(0);
//             std::vector<float> next_hidden_vec(next_hidden.data_ptr<float>(),
//                                                next_hidden.data_ptr<float>() + hidden_dim);
//             node->children[a] = std::make_shared<MCTSNode>(a, priors[a], next_hidden_vec);
//         }
//     }
//
//     float simulate(std::shared_ptr<MCTSNode> node, int depth, int max_depth) {
//         if (depth >= max_depth || node->children.empty()) {
//             return node->value;
//         }
//
//         if (node->visit_count == 0) {
//             expand_node(node);
//         }
//
//         auto best_child = node->children.begin()->second;
//         float best_ucb = -std::numeric_limits<float>::infinity();
//         for (const auto& [action, child] : node->children) {
//             float ucb = ucb_score(child, node->visit_count);
//             if (ucb > best_ucb) {
//                 best_ucb = ucb;
//                 best_child = child;
//             }
//         }
//
//         float reward = node->reward;
//         float value = simulate(best_child, depth + 1, max_depth);
//         best_child->value += value;
//         best_child->visit_count++;
//         node->visit_count++;
//         return reward + gamma * value;
//     }
//
//     std::pair<int, std::vector<float>> run_mcts(torch::Tensor state) {
//         torch::NoGradGuard no_grad;
//         state = state.to(device);
//         auto hidden = rep_net.forward(state).cpu().squeeze(0);
//         std::vector<float> hidden_vec(hidden.data_ptr<float>(), hidden.data_ptr<float>() + hidden_dim);
//         auto root = std::make_shared<MCTSNode>(-1, 0.0, hidden_vec);
//
//         for (int i = 0; i < num_simulations; ++i) {
//             simulate(root, 0, 5);
//         }
//
//         std::vector<float> visit_counts(action_dim, 0.0);
//         float sum_visits = 0.0;
//         for (const auto& [action, child] : root->children) {
//             visit_counts[action] = static_cast<float>(child->visit_count);
//             sum_visits += child->visit_count;
//         }
//         for (auto& v : visit_counts) v /= sum_visits;
//
//         std::discrete_distribution<int> dist(visit_counts.begin(), visit_counts.end());
//         int action = dist(gen);
//
//         return {action, visit_counts};
//     }
//
//     void train_step(int batch_size, int unroll_steps) {
//         if (replay_buffer.size() < batch_size) return;
//
//         auto traj = replay_buffer.sample(batch_size);
//         int T = traj.states.size();
//         if (T < unroll_steps + 1) return;
//
//         std::vector<float> states;
//         for (int t = 0; t <= unroll_steps; ++t) {
//             states.insert(states.end(), traj.states[t].begin(), traj.states[t].end());
//         }
//
//         auto state_tensor = torch::tensor(states).view({unroll_steps + 1, state_dim}).to(device);
//         auto hidden = rep_net.forward(state_tensor[0].unsqueeze(0)).squeeze(0);
//
//         torch::Tensor policy_loss = torch::zeros({1}).to(device);
//         torch::Tensor value_loss = torch::zeros({1}).to(device);
//         torch::Tensor reward_loss = torch::zeros({1}).to(device);
//
//         for (int k = 0; k < unroll_steps; ++k) {
//             auto [policy, value] = pred_net.forward(hidden.unsqueeze(0));
//             policy_loss += torch::nn::functional::cross_entropy(
//                 policy, torch::tensor(traj.policy_targets[k]).to(device));
//             value_loss += torch::mse_loss(value.squeeze(-1),
//                                           torch::tensor(traj.value_targets[k]).to(device));
//
//             auto action_tensor = torch::zeros({1, action_dim}).to(device);
//             action_tensor[0][traj.actions[k]] = 1.0;
//             hidden = dyn_net.forward(hidden.unsqueeze(0), action_tensor).squeeze(0);
//
//             if (k < unroll_steps - 1) {
//                 reward_loss += torch::mse_loss(
//                     value, torch::tensor(traj.rewards[k]).to(device));
//             }
//         }
//
//         auto loss = (policy_loss + value_loss + reward_loss) / unroll_steps;
//
//         optimizer.zero_grad();
//         loss.backward();
//         torch::nn::utils::clip_grad_norm_({rep_net.parameters(), dyn_net.parameters(), pred_net.parameters()}, 1.0);
//         optimizer.step();
//     }
//
//     void collect_trajectory(CartPoleEnv& env, int max_steps) {
//         ReplayBuffer::Trajectory traj;
//         env.reset();
//         auto state = torch::tensor(env.state).view({1, state_dim});
//
//         for (int t = 0; t < max_steps && !env.done; ++t) {
//             traj.states.push_back(state.view(-1).tolist<float>());
//             auto [action, policy_target] = run_mcts(state);
//             traj.actions.push_back(action);
//             traj.policy_targets.push_back(policy_target);
//
//             auto [next_state, reward, done] = env.step(action);
//             auto next_state_tensor = torch::tensor(next_state).view({1, state_dim});
//             traj.rewards.push_back(reward);
//
//             auto hidden = rep_net.forward(state.to(device)).cpu().squeeze(0);
//             auto [_, value] = pred_net.forward(hidden.unsqueeze(0).to(device));
//             traj.value_targets.push_back(value.cpu().item<float>());
//
//             state = next_state_tensor;
//         }
//
//         // Bootstrap value for the last state
//         if (!env.done) {
//             auto hidden = rep_net.forward(state.to(device)).cpu().squeeze(0);
//             auto [_, value] = pred_net.forward(hidden.unsqueeze(0).to(device));
//             traj.value_targets.back() = value.cpu().item<float>();
//         } else {
//             traj.value_targets.back() = 0.0;
//         }
//
//         replay_buffer.push(traj);
//     }
// };
//
// int main() {
//     torch::manual_seed(42);
//     std::mt19937 gen(std::random_device{}());
//
//     // Environment and agent
//     CartPoleEnv env;
//     MuZeroAgent agent(
//         4, 2, 64, 0.99, 0.001, 100, 50, 0.25, 0.25 // state_dim, action_dim, hidden_dim, gamma, lr, buffer_capacity, num_simulations, dirichlet_alpha, exploration_fraction
//     );
//
//     // Training loop
//     const int episodes = 500;
//     const int batch_size = 1;
//     const int max_steps = 200;
//     const int unroll_steps = 5;
//
//     for (int episode = 0; episode < episodes; ++episode) {
//         agent.collect_trajectory(env, max_steps);
//         agent.train_step(batch_size, unroll_steps);
//
//         float episode_reward = 0.0;
//         int steps = 0;
//         env.reset();
//         auto state = torch::tensor(env.state).view({1, state_dim});
//
//         while (!env.done && steps < max_steps) {
//             auto [action, _] = agent.run_mcts(state);
//             auto [next_state, reward, done] = env.step(action);
//             state = torch::tensor(next_state).view({1, state_dim});
//             episode_reward += reward;
//             steps++;
//         }
//
//         if (episode % 10 == 0) {
//             std::cout << "Episode: " << episode << ", Reward: " << episode_reward
//                       << ", Steps: " << steps << std::endl;
//         }
//     }
//
//     // Save networks
//     torch::save(agent.rep_net, "muzero_rep_net.pt");
//     torch::save(agent.dyn_net, "muzero_dyn_net.pt");
//     torch::save(agent.pred_net, "muzero_pred_net.pt");
//
//     return 0;
// }


namespace xt::models
{
    MuZero::MuZero(int num_classes, int in_channels)
    {
    }

    MuZero::MuZero(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void MuZero::reset()
    {
    }

    auto MuZero::forward(std::initializer_list<std::any> tensors) -> std::any
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
