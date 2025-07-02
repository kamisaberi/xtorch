#include "include/models/reinforcement_learning/pro.h"


using namespace std;

// #include <torch/torch.h>
// #include <vector>
// #include <random>
// #include <iostream>
// #include <deque>
// #include <cmath>
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
// // Trajectory Buffer
// struct TrajectoryBuffer {
//     struct Transition {
//         std::vector<float> state;
//         int action;
//         float reward, value, log_prob, gae, return_;
//     };
//
//     std::vector<Transition> buffer;
//     size_t capacity;
//
//     TrajectoryBuffer(size_t capacity_) : capacity(capacity_) {}
//
//     void push(const std::vector<float>& state, int action, float reward, float value, float log_prob) {
//         if (buffer.size() >= capacity) {
//             buffer.clear();
//         }
//         buffer.push_back({state, action, reward, value, log_prob, 0.0, 0.0});
//     }
//
//     void compute_returns_and_advantages(float gamma, float lambda) {
//         float next_return = 0.0;
//         for (int t = buffer.size() - 1; t >= 0; --t) {
//             auto& trans = buffer[t];
//             float delta = trans.reward + gamma * next_return - trans.value;
//             trans.gae = delta + gamma * lambda * (t < static_cast<int>(buffer.size()) - 1 ? buffer[t + 1].gae : 0.0);
//             trans.return_ = trans.gae + trans.value;
//             next_return = trans.return_;
//         }
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_batch() {
//         std::vector<float> states, rewards, values, log_probs;
//         std::vector<int64_t> actions;
//
//         for (const auto& trans : buffer) {
//             states.insert(states.end(), trans.state.begin(), trans.state.end());
//             actions.push_back(trans.action);
//             rewards.push_back(trans.return_);
//             values.push_back(trans.gae);
//             log_probs.push_back(trans.log_prob);
//         }
//
//         auto state_tensor = torch::tensor(states).view({static_cast<int64_t>(buffer.size()), 4});
//         auto action_tensor = torch::tensor(actions).view({static_cast<int64_t>(buffer.size()), 1});
//         auto return_tensor = torch::tensor(rewards).view({static_cast<int64_t>(buffer.size()), 1});
//         auto advantage_tensor = torch::tensor(values).view({static_cast<int64_t>(buffer.size()), 1});
//         auto log_prob_tensor = torch::tensor(log_probs).view({static_cast<int64_t>(buffer.size()), 1});
//
//         return {state_tensor, action_tensor, return_tensor, advantage_tensor, log_prob_tensor};
//     }
//
//     void clear() { buffer.clear(); }
// };
//
// // Actor-Critic Network
// struct ActorCriticNetwork : torch::nn::Module {
//     torch::nn::Linear fc_common{nullptr}, fc_policy{nullptr}, fc_value{nullptr};
//
//     ActorCriticNetwork(int state_dim, int action_dim) {
//         fc_common = register_module("fc_common", torch::nn::Linear(state_dim, 128));
//         fc_policy = register_module("fc_policy", torch::nn::Linear(128, action_dim));
//         fc_value = register_module("fc_value", torch::nn::Linear(128, 1));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         x = torch::relu(fc_common->forward(x));
//         auto policy_logits = fc_policy->forward(x);
//         auto policy = torch::softmax(policy_logits, -1);
//         auto value = fc_value->forward(x);
//         return {policy, value};
//     }
// };
//
// // PPO Agent
// class PPOAgent {
// public:
//     ActorCriticNetwork policy_net;
//     TrajectoryBuffer buffer;
//     torch::optim::Adam optimizer;
//     torch::Device device;
//     float gamma, lambda, clip_epsilon, value_weight, entropy_weight;
//     int state_dim, action_dim, epochs, batch_size;
//     std::mt19937 gen;
//
//     PPOAgent(int state_dim_, int action_dim_, float gamma_, float lambda_, float clip_epsilon_,
//              float lr_, size_t buffer_capacity_, int epochs_, int batch_size_,
//              float value_weight_, float entropy_weight_)
//         : policy_net(state_dim_, action_dim_),
//           buffer(buffer_capacity_),
//           optimizer(policy_net.parameters(), torch::optim::AdamOptions(lr_)),
//           device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
//           gamma(gamma_),
//           lambda(lambda_),
//           clip_epsilon(clip_epsilon_),
//           value_weight(value_weight_),
//           entropy_weight(entropy_weight_),
//           state_dim(state_dim_),
//           action_dim(action_dim_),
//           epochs(epochs_),
//           batch_size(batch_size_),
//           gen(std::random_device{}()) {
//         policy_net.to(device);
//     }
//
//     int choose_action(torch::Tensor state, float& log_prob_out, float& value_out) {
//         policy_net.eval();
//         torch::NoGradGuard no_grad;
//         state = state.to(device);
//         auto [policy, value] = policy_net.forward(state);
//         value_out = value.cpu().item<float>();
//
//         auto probs = policy.squeeze(0).cpu();
//         auto action_dist = torch::distributions::Categorical(probs);
//         int action = action_dist.sample().item<int>();
//         log_prob_out = action_dist.log_prob(torch::tensor(action)).item<float>();
//
//         policy_net.train();
//         return action;
//     }
//
//     void train_step() {
//         buffer.compute_returns_and_advantages(gamma, lambda);
//         auto [states, actions, returns, advantages, old_log_probs] = buffer.get_batch();
//         states = states.to(device);
//         actions = actions.to(device);
//         returns = returns.to(device);
//         advantages = advantages.to(device);
//         old_log_probs = old_log_probs.to(device);
//
//         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);
//
//         for (int epoch = 0; epoch < epochs; ++epoch) {
//             for (int i = 0; i < states.size(0); i += batch_size) {
//                 auto idx = torch::arange(i, std::min(i + batch_size, static_cast<int>(states.size(0))));
//                 auto state_batch = states.index_select(0, idx);
//                 auto action_batch = actions.index_select(0, idx);
//                 auto return_batch = returns.index_select(0, idx);
//                 auto advantage_batch = advantages.index_select(0, idx);
//                 auto old_log_prob_batch = old_log_probs.index_select(0, idx);
//
//                 auto [policy, value] = policy_net.forward(state_batch);
//                 auto log_probs = torch::log_softmax(policy, -1).gather(1, action_batch);
//                 auto ratio = torch::exp(log_probs - old_log_prob_batch);
//                 auto surr1 = ratio * advantage_batch;
//                 auto surr2 = torch::clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage_batch;
//                 auto policy_loss = -torch::min(surr1, surr2).mean();
//                 auto value_loss = torch::mse_loss(value, return_batch);
//                 auto entropy = -(policy * torch::log_softmax(policy, -1)).sum(-1).mean();
//                 auto loss = policy_loss + value_weight * value_loss - entropy_weight * entropy;
//
//                 optimizer.zero_grad();
//                 loss.backward();
//                 torch::nn::utils::clip_grad_norm_(policy_net.parameters(), 1.0);
//                 optimizer.step();
//             }
//         }
//
//         buffer.clear();
//     }
//
//     void collect_trajectory(CartPoleEnv& env, int max_steps) {
//         env.reset();
//         auto state = torch::tensor(env.state).view({1, state_dim});
//         float episode_reward = 0.0;
//         int steps = 0;
//
//         while (!env.done && steps < max_steps) {
//             float log_prob, value;
//             int action = choose_action(state, log_prob, value);
//             auto [next_state, reward, done] = env.step(action);
//             auto next_state_tensor = torch::tensor(next_state).view({1, state_dim});
//             buffer.push(state.view(-1).tolist<float>(), action, reward, value, log_prob);
//             episode_reward += reward;
//             steps++;
//             state = next_state_tensor;
//         }
//
//         return {episode_reward, steps};
//     }
// };
//
// int main() {
//     torch::manual_seed(42);
//     std::mt19937 gen(std::random_device{}());
//
//     // Environment and agent
//     CartPoleEnv env;
//     PPOAgent agent(
//         4, 2, 0.99, 0.95, 0.2, 0.001, 2048, 10, 64, 0.5, 0.01 // state_dim, action_dim, gamma, lambda, clip_epsilon, lr, buffer_capacity, epochs, batch_size, value_weight, entropy_weight
//     );
//
//     // Training loop
//     const int episodes = 500;
//     const int max_steps = 200;
//
//     for (int episode = 0; episode < episodes; ++episode) {
//         auto [episode_reward, steps] = agent.collect_trajectory(env, max_steps);
//         agent.train_step();
//
//         if (episode % 10 == 0) {
//             std::cout << "Episode: " << episode << ", Reward: " << episode_reward
//                       << ", Steps: " << steps << std::endl;
//         }
//     }
//
//     // Save policy network
//     torch::save(agent.policy_net, "ppo_policy.pt");
//
//     return 0;
// }
//


namespace xt::models
{
    PPO::PPO(int num_classes, int in_channels)
    {
    }

    PPO::PPO(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void PPO::reset()
    {
    }

    auto PPO::forward(std::initializer_list<std::any> tensors) -> std::any
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
