#include <models/reinforcement_learning/rainbow.h>


using namespace std;

// #include <torch/torch.h>
// #include <vector>
// #include <random>
// #include <iostream>
// #include <deque>
// #include <cmath>
// #include <algorithm>
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
// // Prioritized Replay Buffer
// struct PrioritizedReplayBuffer {
//     struct Transition {
//         std::vector<float> state, next_state;
//         int action;
//         float reward;
//         bool done;
//         float priority;
//     };
//
//     std::deque<Transition> buffer;
//     std::vector<float> priorities;
//     size_t capacity;
//     float alpha, beta, beta_increment;
//     std::mt19937 gen;
//
//     PrioritizedReplayBuffer(size_t capacity_, float alpha_, float beta_, float beta_increment_)
//         : capacity(capacity_), alpha(alpha_), beta(beta_), beta_increment(beta_increment_),
//           gen(std::random_device{}()) {}
//
//     void push(const std::vector<float>& state, int action, float reward,
//               const std::vector<float>& next_state, bool done, float priority) {
//         if (buffer.size() >= capacity) {
//             buffer.pop_front();
//             priorities.erase(priorities.begin());
//         }
//         buffer.push_back({state, next_state, action, reward, done, priority});
//         priorities.push_back(std::pow(priority, alpha));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<size_t>>
//     sample(size_t batch_size) {
//         std::vector<float> probs = priorities;
//         float sum_probs = std::accumulate(probs.begin(), probs.end(), 0.0f);
//         for (auto& p : probs) p /= sum_probs;
//
//         std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
//         std::vector<size_t> indices;
//         std::vector<float> states, next_states, rewards, dones, weights;
//         std::vector<int64_t> actions;
//
//         for (size_t i = 0; i < batch_size; ++i) {
//             size_t idx = dist(gen);
//             indices.push_back(idx);
//             auto& t = buffer[idx];
//             states.insert(states.end(), t.state.begin(), t.state.end());
//             next_states.insert(next_states.end(), t.next_state.begin(), t.next_state.end());
//             actions.push_back(t.action);
//             rewards.push_back(t.reward);
//             dones.push_back(static_cast<float>(t.done));
//             float weight = std::pow(buffer.size() * probs[idx], -beta);
//             weights.push_back(weight);
//         }
//
//         beta = std::min(1.0f, beta + beta_increment);
//
//         auto state_tensor = torch::tensor(states).view({static_cast<int64_t>(batch_size), 4});
//         auto action_tensor = torch::tensor(actions).view({static_cast<int64_t>(batch_size), 1});
//         auto reward_tensor = torch::tensor(rewards).view({static_cast<int64_t>(batch_size), 1});
//         auto next_state_tensor = torch::tensor(next_states).view({static_cast<int64_t>(batch_size), 4});
//         auto done_tensor = torch::tensor(dones).view({static_cast<int64_t>(batch_size), 1});
//         auto weight_tensor = torch::tensor(weights).view({static_cast<int64_t>(batch_size), 1});
//
//         return {state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor, weight_tensor, indices};
//     }
//
//     void update_priorities(const std::vector<size_t>& indices, const std::vector<float>& errors) {
//         for (size_t i = 0; i < indices.size(); ++i) {
//             buffer[indices[i]].priority = errors[i] + 1e-5;
//             priorities[indices[i]] = std::pow(buffer[indices[i]].priority, alpha);
//         }
//     }
//
//     size_t size() const { return buffer.size(); }
// };
//
// // Distributional Dueling Network with Noisy Layers
// struct NoisyLinear : torch::nn::Module {
//     torch::nn::Linear mu_w{nullptr}, mu_b{nullptr};
//     torch::Tensor sigma_w, sigma_b, epsilon_w, epsilon_b;
//     int in_features, out_features;
//     float sigma_init;
//
//     NoisyLinear(int in_features_, int out_features_, float sigma_init_)
//         : in_features(in_features_), out_features(out_features_), sigma_init(sigma_init_) {
//         mu_w = register_module("mu_w", torch::nn::Linear(in_features, out_features));
//         mu_b = register_module("mu_b", torch::nn::Linear(1, out_features));
//         sigma_w = register_buffer("sigma_w", torch::full({out_features, in_features}, sigma_init));
//         sigma_b = register_buffer("sigma_b", torch::full({out_features}, sigma_init));
//         reset_noise();
//     }
//
//     void reset_noise() {
//         auto epsilon_in = torch::randn({in_features, 1}) / std::sqrt(static_cast<float>(in_features));
//         auto epsilon_out = torch::randn({out_features, 1}) / std::sqrt(static_cast<float>(out_features));
//         epsilon_w = torch::sign(epsilon_in) * torch::sqrt(torch::abs(epsilon_in)) *
//                     torch::sign(epsilon_out).t() * torch::sqrt(torch::abs(epsilon_out)).t();
//         epsilon_b = torch::sign(epsilon_out) * torch::sqrt(torch::abs(epsilon_out));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         return mu_w->forward(x) + mu_b->forward(torch::ones({x.size(0), 1})) +
//                (sigma_w * epsilon_w).matmul(x) + (sigma_b * epsilon_b).squeeze(-1);
//     }
// };
//
// struct DistributionalDuelingNetwork : torch::nn::Module {
//     NoisyLinear fc_common{nullptr}, value_stream{nullptr}, advantage_stream{nullptr};
//     int num_atoms;
//     float v_min, v_max;
//
//     DistributionalDuelingNetwork(int state_dim, int action_dim, int num_atoms_, float v_min_, float v_max_)
//         : num_atoms(num_atoms_), v_min(v_min_), v_max(v_max_) {
//         fc_common = register_module("fc_common", NoisyLinear(state_dim, 128, 0.1f));
//         value_stream = register_module("value_stream", NoisyLinear(128, num_atoms));
//         advantage_stream = register_module("advantage_stream", NoisyLinear(128, action_dim * num_atoms));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(fc_common->forward(x));
//         auto value = value_stream->forward(x).view({-1, 1, num_atoms});
//         auto advantage = advantage_stream->forward(x).view({-1, -2, num_atoms});
//         auto q_dist = value + (advantage - advantage.mean(1, true));
//         return torch::softmax(q_dist, -1); // Probabilities over atoms
//     }
//
//     void reset_noise() {
//         fc_common->reset_noise();
//         value_stream->reset_noise();
//         advantage_stream->reset_noise();
//     }
// };
//
// // Rainbow Agent
// class RainbowAgent {
// public:
//     DistributionalDuelingNetwork policy_net, target_net;
//     PrioritizedReplayBuffer replay_buffer;
//     torch::optim::Adam optimizer;
//     torch::Device device;
//     float gamma, eta, n_steps;
//     int state_dim, action_dim, num_atoms, target_update_freq, step_count;
//     std::vector<float> atoms;
//     std::mt19937 gen;
//
//     RainbowAgent(int state_dim_, int action_dim_, float gamma_, float eta_, float lr_,
//                  size_t buffer_capacity_, float beta_, int n_steps_, int num_atoms_,
//                  float v_min_, float v_max_, int target_update_freq_)
//         : policy_net(state_dim_, action_dim, num_atoms_, v_min_, v_max_),
//           target_net(state_dim_, action_dim, num_atoms_, v_min_, v_max_),
//           replay_buffer(buffer_capacity_, 0.6f, beta_, 0.0001f),
//           optimizer(policy_net.parameters(), torch::optim::AdamOptions(lr_)),
//           device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
//           gamma(gamma_),
//           eta(eta_),
//           n_steps(n_steps_),
//           state_dim(state_dim_),
//           action_dim(action_dim_),
//           num_atoms(num_atoms_),
//           target_update_freq(target_update_freq_),
//           step_count(0),
//           gen(std::random_device{}()) {
//         policy_net.to(device);
//         target_net.to(device);
//         target_net.load_state_dict(policy_net.state_dict());
//         target_net.eval();
//         atoms = std::vector<float>(num_atoms);
//         float delta = (v_max_ - v_min_) / (num_atoms - 1);
//         for (int i = 0; i < num_atoms; ++i) {
//             atoms[i] = v_min_ + i * delta;
//         }
//     }
//
//     int select_action(torch::Tensor state) {
//         policy_net.eval();
//         torch::NoGradGuard no_grad;
//         state = state.to(device);
//         auto q_dist = policy_net.forward(state); // [1, action_dim, num_atoms]
//         auto q_values = (q_dist * torch::tensor(atoms).to(device)).sum(-1); // [1, action_dim]
//         policy_net.reset_noise();
//         policy_net.train();
//         return q_values.argmax().item<int64_t>();
//     }
//
//     void train_step(int batch_size) {
//         if (replay_buffer.size() < batch_size) return;
//
//         auto [states, actions, rewards, next_states, dones, weights, indices] =
//             replay_buffer.sample(batch_size);
//         states = states.to(device);
//         actions = actions.to(device);
//         rewards = rewards.to(device);
//         next_states = next_states.to(device);
//         dones = dones.to(device);
//         weights = weights.to(device);
//
//         // Current Q distribution
//         auto q_dist = policy_net.forward(states); // [batch, action_dim, num_atoms]
//         q_dist = q_dist.index_select(1, actions.squeeze(-1)).diagonal().t(); // [batch, num_atoms]
//
//         // Double DQN: policy net selects actions, target net evaluates
//         torch::NoGradGuard no_grad;
//         auto next_q_dist = policy_net.forward(next_states); // [batch, action_dim, num_atoms]
//         auto next_actions = (next_q_dist * torch::tensor(atoms).to(device)).sum(-1).argmax(1, true);
//         auto target_q_dist = target_net.forward(next_states); // [batch, action_dim, num_atoms]
//         target_q_dist = target_q_dist.index_select(1, next_actions.squeeze(-1)).diagonal().t(); // [batch, num_atoms]
//
//         // n-step returns
//         auto target_atoms = rewards + std::pow(gamma, n_steps) * (1 - dones) * torch::tensor(atoms).to(device);
//         target_atoms = torch::clamp(target_atoms, policy_net->v_min, policy_net->v_max);
//
//         // Project target distribution
//         auto delta_z = (policy_net->v_max - policy_net->v_min) / (num_atoms - 1);
//         auto b = (target_atoms - policy_net->v_min) / delta_z;
//         auto l = torch::floor(b).to(torch::kInt64);
//         auto u = torch::ceil(b).to(torch::kInt64);
//         l = torch::clamp(l, 0, num_atoms - 1);
//         u = torch::clamp(u, 0, num_atoms - 1);
//
//         auto m = torch::zeros({batch_size, num_atoms}).to(device);
//         for (int i = 0; i < batch_size; ++i) {
//             for (int j = 0; j < num_atoms; ++j) {
//                 m[i][l[i][j].item<int64_t>()] += target_q_dist[i][j] * (u[i][j] - b[i][j]);
//                 m[i][u[i][j].item<int64_t>()] += target_q_dist[i][j] * (b[i][j] - l[i][j]);
//             }
//         }
//
//         // Loss: KL divergence with importance sampling weights
//         auto log_q_dist = torch::log(q_dist + 1e-10);
//         auto loss = -(m * log_q_dist).sum(-1) * weights;
//         loss = loss.mean() * eta; // Scale by distributional loss weight
//
//         // Optimize
//         optimizer.zero_grad();
//         loss.backward();
//         torch::nn::utils::clip_grad_norm_(policy_net.parameters(), 1.0);
//         optimizer.step();
//
//         // Update priorities
//         auto td_errors = (q_dist - m).abs().sum(-1).cpu().data_ptr<float>();
//         std::vector<float> errors(td_errors, td_errors + batch_size);
//         replay_buffer.update_priorities(indices, errors);
//
//         // Update target network
//         step_count++;
//         if (step_count % target_update_freq == 0) {
//             target_net.load_state_dict(policy_net.state_dict());
//         }
//     }
// };
//
// int main() {
//     torch::manual_seed(42);
//     std::mt19937 gen(std::random_device{}());
//
//     // Environment and agent
//     CartPoleEnv env;
//     RainbowAgent agent(
//         4, 2, 0.99, 0.5, 0.001, 10000, 0.4, 3, 51, -10.0, 10.0, 100 // state_dim, action_dim, gamma, eta, lr, buffer_capacity, beta, n_steps, num_atoms, v_min, v_max, target_update_freq
//     );
//
//     // Training loop
//     const int episodes = 500;
//     const int batch_size = 32;
//
//     for (int episode = 0; episode < episodes; ++episode) {
//         env.reset();
//         auto state = torch::tensor(env.state).view({1, 4});
//         float episode_reward = 0.0;
//         int steps = 0;
//
//         while (!env.done) {
//             // Select action
//             int action = agent.select_action(state);
//
//             // Step environment
//             auto [next_state, reward, done] = env.step(action);
//             auto next_state_tensor = torch::tensor(next_state).view({1, 4});
//             episode_reward += reward;
//             steps++;
//
//             // Store transition with initial priority
//             float priority = 1.0; // Simplified: use max priority for new transitions
//             agent.replay_buffer.push(state.view(-1).tolist<float>(), action, reward,
//                                      next_state_tensor.view(-1).tolist<float>(), done, priority);
//
//             // Train
//             agent.train_step(batch_size);
//
//             // Update state
//             state = next_state_tensor;
//
//             if (done) break;
//         }
//
//         if (episode % 10 == 0) {
//             std::cout << "Episode: " << episode << ", Reward: " << episode_reward
//                       << ", Steps: " << steps << ", Buffer Size: " << agent.replay_buffer.size() << std::endl;
//         }
//     }
//
//     // Save policy network
//     torch::save(agent.policy_net, "rainbow_policy.pt");
//
//     return 0;
// }


namespace xt::models
{
    Rainbow::Rainbow(int num_classes, int in_channels)
    {
    }

    Rainbow::Rainbow(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void Rainbow::reset()
    {
    }

    auto Rainbow::forward(std::initializer_list<std::any> tensors) -> std::any
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
