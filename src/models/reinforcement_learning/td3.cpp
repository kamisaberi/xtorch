#include "include/models/reinforcement_learning/td3.h"


using namespace std;

// #include <torch/torch.h>
// #include <vector>
// #include <random>
// #include <iostream>
// #include <deque>
// #include <cmath>
//
// // Simulated Pendulum Environment
// struct PendulumEnv {
//     std::vector<float> state;
//     float gravity = 9.8, mass = 1.0, length = 1.0, max_torque = 2.0;
//     float dt = 0.05, max_speed = 8.0, max_angle = M_PI;
//     bool done = false;
//     int steps = 0;
//     const int max_steps = 200;
//     std::mt19937 gen;
//
//     PendulumEnv() : state(3, 0.0), gen(std::random_device{}()) {
//         reset();
//     }
//
//     void reset() {
//         std::uniform_real_distribution<float> dist(-0.05, 0.05);
//         state = {std::cos(M_PI + dist(gen)), std::sin(M_PI + dist(gen)), dist(gen)};
//         done = false;
//         steps = 0;
//     }
//
//     std::tuple<std::vector<float>, float, bool> step(float action) {
//         action = std::clamp(action, -max_torque, max_torque);
//         float theta = std::atan2(state[1], state[0]);
//         float theta_dot = state[2];
//
//         float acc = (-gravity / length * std::sin(theta) + action / (mass * length * length)) /
//                     (4.0 / 3.0);
//         theta_dot += acc * dt;
//         theta_dot = std::clamp(theta_dot, -max_speed, max_speed);
//         theta += theta_dot * dt;
//
//         state = {std::cos(theta), std::sin(theta), theta_dot};
//         steps++;
//
//         float reward = -(theta * theta + 0.1 * theta_dot * theta_dot + 0.001 * action * action);
//         done = (steps >= max_steps);
//
//         return {state, reward, done};
//     }
// };
//
// // Replay Buffer
// struct ReplayBuffer {
//     struct Transition {
//         std::vector<float> state, next_state;
//         float action, reward;
//         bool done;
//     };
//
//     std::deque<Transition> buffer;
//     size_t capacity;
//     std::mt19937 gen;
//
//     ReplayBuffer(size_t capacity_) : capacity(capacity_), gen(std::random_device{}()) {}
//
//     void push(const std::vector<float>& state, float action, float reward,
//               const std::vector<float>& next_state, bool done) {
//         if (buffer.size() >= capacity) {
//             buffer.pop_front();
//         }
//         buffer.push_back({state, next_state, action, reward, done});
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(size_t batch_size) {
//         std::uniform_int_distribution<size_t> dist(0, buffer.size() - 1);
//         std::vector<float> states, next_states, actions, rewards, dones;
//
//         for (size_t i = 0; i < batch_size; ++i) {
//             size_t idx = dist(gen);
//             auto& t = buffer[idx];
//             states.insert(states.end(), t.state.begin(), t.state.end());
//             next_states.insert(next_states.end(), t.next_state.begin(), t.next_state.end());
//             actions.push_back(t.action);
//             rewards.push_back(t.reward);
//             dones.push_back(static_cast<float>(t.done));
//         }
//
//         auto state_tensor = torch::tensor(states).view({static_cast<int64_t>(batch_size), 3});
//         auto action_tensor = torch::tensor(actions).view({static_cast<int64_t>(batch_size), 1});
//         auto reward_tensor = torch::tensor(rewards).view({static_cast<int64_t>(batch_size), 1});
//         auto next_state_tensor = torch::tensor(next_states).view({static_cast<int64_t>(batch_size), 3});
//         auto done_tensor = torch::tensor(dones).view({static_cast<int64_t>(batch_size), 1});
//
//         return {state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor};
//     }
//
//     size_t size() const { return buffer.size(); }
// };
//
// // Actor Network
// struct ActorNetwork : torch::nn::Module {
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
//
//     ActorNetwork(int state_dim, int action_dim, float max_action) : max_action(max_action) {
//         fc1 = register_module("fc1", torch::nn::Linear(state_dim, 128));
//         fc2 = register_module("fc2", torch::nn::Linear(128, 64));
//         fc3 = register_module("fc3", torch::nn::Linear(64, action_dim));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(fc1->forward(x));
//         x = torch::relu(fc2->forward(x));
//         return max_action * torch::tanh(fc3->forward(x));
//     }
//
// private:
//     float max_action;
// };
//
// // Critic Network
// struct CriticNetwork : torch::nn::Module {
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
//
//     CriticNetwork(int state_dim, int action_dim) {
//         fc1 = register_module("fc1", torch::nn::Linear(state_dim + action_dim, 128));
//         fc2 = register_module("fc2", torch::nn::Linear(128, 64));
//         fc3 = register_module("fc3", torch::nn::Linear(64, 1));
//     }
//
//     torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
//         auto x = torch::cat({state, action}, -1);
//         x = torch::relu(fc1->forward(x));
//         x = torch::relu(fc2->forward(x));
//         return fc3->forward(x);
//     }
// };
//
// // TD3 Agent
// class TD3Agent {
// public:
//     ActorNetwork actor_net, actor_target_net;
//     CriticNetwork q1_net, q2_net, q1_target_net, q2_target_net;
//     ReplayBuffer replay_buffer;
//     torch::optim::Adam actor_optimizer, critic_optimizer;
//     torch::Device device;
//     float gamma, tau, policy_noise, noise_clip, policy_delay;
//     int state_dim, action_dim, batch_size, update_freq, step_count;
//     float max_action;
//     std::mt19937 gen;
//
//     TD3Agent(int state_dim_, int action_dim_, float max_action_, float gamma_, float tau_, float lr_,
//              float policy_noise_, float noise_clip_, int policy_delay_, size_t buffer_capacity_,
//              int batch_size_, int update_freq_)
//         : actor_net(state_dim_, action_dim_, max_action_),
//           actor_target_net(state_dim_, action_dim_, max_action_),
//           q1_net(state_dim_, action_dim_),
//           q2_net(state_dim_, action_dim_),
//           q1_target_net(state_dim_, action_dim_),
//           q2_target_net(state_dim_, action_dim_),
//           replay_buffer(buffer_capacity_),
//           actor_optimizer(actor_net.parameters(), torch::optim::AdamOptions(lr_)),
//           critic_optimizer({q1_net.parameters(), q2_net.parameters()}, torch::optim::AdamOptions(lr_)),
//           device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
//           gamma(gamma_),
//           tau(tau_),
//           policy_noise(policy_noise_),
//           noise_clip(noise_clip_),
//           policy_delay(policy_delay_),
//           state_dim(state_dim_),
//           action_dim(action_dim_),
//           batch_size(batch_size_),
//           update_freq(update_freq_),
//           step_count(0),
//           max_action(max_action_),
//           gen(std::random_device{}()) {
//         actor_net.to(device);
//         actor_target_net.to(device);
//         q1_net.to(device);
//         q2_net.to(device);
//         q1_target_net.to(device);
//         q2_target_net.to(device);
//         actor_target_net.load_state_dict(actor_net.state_dict());
//         q1_target_net.load_state_dict(q1_net.state_dict());
//         q2_target_net.load_state_dict(q2_net.state_dict());
//         actor_target_net.eval();
//         q1_target_net.eval();
//         q2_target_net.eval();
//     }
//
//     float select_action(torch::Tensor state, bool add_noise = true) {
//         actor_net.eval();
//         torch::NoGradGuard no_grad;
//         state = state.to(device);
//         auto action = actor_net.forward(state).cpu().item<float>();
//         actor_net.train();
//
//         if (add_noise) {
//             std::normal_distribution<float> noise_dist(0.0, policy_noise);
//             float noise = noise_dist(gen);
//             noise = std::clamp(noise, -noise_clip, noise_clip);
//             action = std::clamp(action + noise, -max_action, max_action);
//         }
//
//         return action;
//     }
//
//     void update_target_networks() {
//         for (auto& param : actor_target_net.named_parameters()) {
//             auto target_param = actor_target_net.named_parameters()[param.key()];
//             target_param.value().copy_(tau * param.value() + (1 - tau) * target_param.value());
//         }
//         for (auto& param : q1_target_net.named_parameters()) {
//             auto target_param = q1_target_net.named_parameters()[param.key()];
//             target_param.value().copy_(tau * param.value() + (1 - tau) * target_param.value());
//         }
//         for (auto& param : q2_target_net.named_parameters()) {
//             auto target_param = q2_target_net.named_parameters()[param.key()];
//             target_param.value().copy_(tau * param.value() + (1 - tau) * target_param.value());
//         }
//     }
//
//     void train_step() {
//         if (replay_buffer.size() < batch_size) return;
//
//         auto [states, actions, rewards, next_states, dones] = replay_buffer.sample(batch_size);
//         states = states.to(device);
//         actions = actions.to(device);
//         rewards = rewards.to(device);
//         next_states = next_states.to(device);
//         dones = dones.to(device);
//
//         // Critic Update
//         torch::NoGradGuard no_grad;
//         auto next_actions = actor_target_net.forward(next_states);
//         auto noise = torch::clamp(torch::randn_like(next_actions) * policy_noise, -noise_clip, noise_clip);
//         next_actions = torch::clamp(next_actions + noise, -max_action, max_action);
//         auto q1_next = q1_target_net.forward(next_states, next_actions);
//         auto q2_next = q2_target_net.forward(next_states, next_actions);
//         auto target_q = rewards + gamma * (1 - dones) * torch::min(q1_next, q2_next);
//
//         auto q1_current = q1_net.forward(states, actions);
//         auto q2_current = q2_net.forward(states, actions);
//         auto critic_loss = torch::mse_loss(q1_current, target_q) + torch::mse_loss(q2_current, target_q);
//
//         critic_optimizer.zero_grad();
//         critic_loss.backward();
//         torch::nn::utils::clip_grad_norm_({q1_net.parameters(), q2_net.parameters()}, 1.0);
//         critic_optimizer.step();
//
//         // Actor Update (delayed)
//         if (step_count % policy_delay == 0) {
//             auto pred_actions = actor_net.forward(states);
//             auto actor_loss = -q1_net.forward(states, pred_actions).mean();
//
//             actor_optimizer.zero_grad();
//             actor_loss.backward();
//             torch::nn::utils::clip_grad_norm_(actor_net.parameters(), 1.0);
//             actor_optimizer.step();
//
//             update_target_networks();
//         }
//     }
//
//     std::pair<float, int> collect_trajectory(PendulumEnv& env, int max_steps) {
//         env.reset();
//         auto state = torch::tensor(env.state).view({1, state_dim});
//         float episode_reward = 0.0;
//         int steps = 0;
//
//         while (!env.done && steps < max_steps) {
//             float action = select_action(state, true);
//             auto [next_state, reward, done] = env.step(action);
//             auto next_state_tensor = torch::tensor(next_state).view({1, state_dim});
//             replay_buffer.push(state.view(-1).tolist<float>(), action, reward,
//                                next_state_tensor.view(-1).tolist<float>(), done);
//             episode_reward += reward;
//             steps++;
//
//             if (replay_buffer.size() >= batch_size && step_count % update_freq == 0) {
//                 train_step();
//             }
//
//             state = next_state_tensor;
//             step_count++;
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
//     PendulumEnv env;
//     TD3Agent agent(
//         3, 1, 2.0, 0.99, 0.005, 0.001, 0.2, 0.4, 2, 10000, 64, 4 // state_dim, action_dim, max_action, gamma, tau, lr, policy_noise, noise_clip, policy_delay, buffer_capacity, batch_size, update_freq
//     );
//
//     // Training loop
//     const int episodes = 500;
//     const int max_steps = 200;
//
//     for (int episode = 0; episode < episodes; ++episode) {
//         auto [episode_reward, steps] = agent.collect_trajectory(env, max_steps);
//
//         if (episode % 10 == 0) {
//             std::cout << "Episode: " << episode << ", Reward: " << episode_reward
//                       << ", Steps: " << steps << std::endl;
//         }
//     }
//
//     // Save networks
//     torch::save(agent.actor_net, "td3_actor.pt");
//     torch::save(agent.q1_net, "td3_q1.pt");
//     torch::save(agent.q2_net, "td3_q2.pt");
//
//     return 0;
// }


namespace xt::models
{
    TD3::TD3(int num_classes, int in_channels)
    {
    }

    TD3::TD3(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void TD3::reset()
    {
    }

    auto TD3::forward(std::initializer_list<std::any> tensors) -> std::any
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
