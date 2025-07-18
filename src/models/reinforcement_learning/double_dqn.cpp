#include <models/reinforcement_learning/double_dqn.h>


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
// // Replay Buffer
// struct ReplayBuffer {
//     struct Transition {
//         std::vector<float> state, next_state;
//         float reward;
//         int action;
//         bool done;
//     };
//
//     std::deque<Transition> buffer;
//     size_t capacity;
//     std::mt19937 gen;
//
//     ReplayBuffer(size_t capacity_) : capacity(capacity_), gen(std::random_device{}()) {}
//
//     void push(const std::vector<float>& state, int action, float reward,
//               const std::vector<float>& next_state, bool done) {
//         if (buffer.size() >= capacity) {
//             buffer.pop_front();
//         }
//         buffer.push_back({state, next_state, reward, action, done});
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(size_t batch_size) {
//         std::uniform_int_distribution<size_t> dist(0, buffer.size() - 1);
//         std::vector<float> states, next_states, rewards, dones;
//         std::vector<int64_t> actions;
//
//         for (size_t i = 0; i < batch_size; ++i) {
//             size_t idx = dist(gen);
//             auto& t = buffer[idx];
//             states.insert(states.end(), t.state.begin(), t.state.end());
//             next_states.insert(next_states.end(), t.next_state.begin(), t.next_state.end());
//             rewards.push_back(t.reward);
//             actions.push_back(t.action);
//             dones.push_back(static_cast<float>(t.done));
//         }
//
//         auto state_tensor = torch::tensor(states).view({static_cast<int64_t>(batch_size), 4});
//         auto action_tensor = torch::tensor(actions).view({static_cast<int64_t>(batch_size), 1});
//         auto reward_tensor = torch::tensor(rewards).view({static_cast<int64_t>(batch_size), 1});
//         auto next_state_tensor = torch::tensor(next_states).view({static_cast<int64_t>(batch_size), 4});
//         auto done_tensor = torch::tensor(dones).view({static_cast<int64_t>(batch_size), 1});
//
//         return {state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor};
//     }
//
//     size_t size() const { return buffer.size(); }
// };
//
// // Q-Network
// struct QNetwork : torch::nn::Module {
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
//
//     QNetwork(int state_dim, int action_dim) {
//         fc1 = register_module("fc1", torch::nn::Linear(state_dim, 128));
//         fc2 = register_module("fc2", torch::nn::Linear(128, 128));
//         fc3 = register_module("fc3", torch::nn::Linear(128, action_dim));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(fc1->forward(x));
//         x = torch::relu(fc2->forward(x));
//         return fc3->forward(x);
//     }
// };
//
// // Double DQN Agent
// class DoubleDQNAgent {
// public:
//     QNetwork policy_net, target_net;
//     ReplayBuffer replay_buffer;
//     torch::optim::Adam optimizer;
//     torch::Device device;
//     float gamma, epsilon, epsilon_min, epsilon_decay;
//     int state_dim, action_dim, target_update_freq, step_count;
//
//     DoubleDQ(int state_dim_, int action_dim_, float gamma_, float epsilon_, float epsilon_min_,
//              float epsilon_decay_, float lr_, size_t buffer_capacity_, int target_update_freq_)
//         : policy_net(state_dim_, action_dim_),
//           target_net(state_dim_, action_dim_),
//           replay_buffer(buffer_capacity_),
//           optimizer(policy_net.parameters(), torch::optim::AdamOptions(lr_)),
//           device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
//           gamma(gamma_),
//           epsilon(epsilon_),
//           epsilon_min(epsilon_min_),
//           epsilon_decay(epsilon_decay_),
//           state_dim(state_dim_),
//           action_dim(action_dim_),
//           target_update_freq(target_update_freq_),
//           step_count(0) {
//         policy_net.to(device);
//         target_net.to(device);
//         target_net.load_state_dict(policy_net.state_dict());
//         target_net.eval();
//     }
//
//     int select_action(torch::Tensor state, std::mt19937& gen) {
//         std::uniform_real_distribution<float> dist(0.0, 1.0);
//         if (dist(gen) < epsilon) {
//             std::uniform_int_distribution<int> action_dist(0, action_dim - 1);
//             return action_dist(gen);
//         } else {
//             policy_net.eval();
//             torch::NoGradGuard no_grad;
//             auto q_values = policy_net.forward(state.to(device));
//             policy_net.train();
//             return q_values.argmax().item<int64_t>();
//         }
//     }
//
//     void update_epsilon() {
//         epsilon = std::max(epsilon_min, epsilon * epsilon_decay);
//     }
//
//     void train_step(int batch_size, std::mt19937& gen) {
//         if (replay_buffer.size() < batch_size) return;
//
//         auto [states, actions, rewards, next_states, dones] = replay_buffer.sample(batch_size);
//         states = states.to(device);
//         actions = actions.to(device);
//         rewards = rewards.to(device);
//         next_states = next_states.to(device);
//         dones = dones.to(device);
//
//         // Compute Q-values
//         auto q_values = policy_net.forward(states).gather(1, actions);
//
//         // Double DQN: Use policy net to select actions, target net to evaluate
//         torch::NoGradGuard no_grad;
//         auto next_actions = policy_net.forward(next_states).argmax(1, true);
//         auto next_q_values = target_net.forward(next_states).gather(1, next_actions);
//         auto target_q = rewards + gamma * next_q_values * (1 - dones);
//
//         // Loss
//         auto loss = torch::mse_loss(q_values, target_q);
//
//         // Optimize
//         optimizer.zero_grad();
//         loss.backward();
//         torch::nn::utils::clip_grad_norm_(policy_net.parameters(), 1.0);
//         optimizer.step();
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
//     DoubleDQNAgent agent(
//         4, 2, 0.99, 1.0, 0.01, 0.995, 0.001, 10000, 100 // state_dim, action_dim, gamma, epsilon, epsilon_min, epsilon_decay, lr, buffer_capacity, target_update_freq
//     );
//
//     // Training loop
//     const int episodes = 500;
//     const int batch_size = 32;
//     for (int episode = 0; episode < episodes; ++episode) {
//         env.reset();
//         auto state = torch::tensor(env.state).view({1, 4});
//         float episode_reward = 0.0;
//         int steps = 0;
//
//         while (!env.done) {
//             // Select action
//             int action = agent.select_action(state, gen);
//
//             // Step environment
//             auto [next_state, reward, done] = env.step(action);
//             auto next_state_tensor = torch::tensor(next_state).view({1, 4});
//             episode_reward += reward;
//             steps++;
//
//             // Store transition
//             agent.replay_buffer.push(state.view(-1).tolist<float>(), action, reward,
//                                      next_state_tensor.view(-1).tolist<float>(), done);
//
//             // Train
//             agent.train_step(batch_size, gen);
//
//             // Update state
//             state = next_state_tensor;
//
//             if (done) break;
//         }
//
//         agent.update_epsilon();
//
//         if (episode % 10 == 0) {
//             std::cout << "Episode: " << episode << ", Reward: " << episode_reward
//                       << ", Steps: " << steps << ", Epsilon: " << agent.epsilon << std::endl;
//         }
//     }
//
//     // Save policy network
//     torch::save(agent.policy_net, "doubledqn_policy.pt");
//
//     return 0;
// }


namespace xt::models
{
    DoubleDQN::DoubleDQN(int num_classes, int in_channels)
    {
    }

    DoubleDQN::DoubleDQN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DoubleDQN::reset()
    {
    }

    auto DoubleDQN::forward(std::initializer_list<std::any> tensors) -> std::any
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
