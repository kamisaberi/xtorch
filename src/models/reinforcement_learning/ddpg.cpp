#include "include/models/reinforcement_learning/ddpg.h"


using namespace std;


#include <torch/torch.h>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

// Simple Pendulum Environment
struct PendulumEnv {
    float theta; // Angle from vertical
    float theta_dot; // Angular velocity
    const float max_theta = M_PI;
    const float max_theta_dot = 8.0;
    const float max_torque = 2.0;
    const float dt = 0.05;
    const float g = 9.81;
    const float m = 1.0;
    const float l = 1.0;

    PendulumEnv() { reset(); }

    torch::Tensor get_state() {
        return torch::tensor({theta, theta_dot}, torch::kFloat32);
    }

    void reset() {
        theta = static_cast<float>(rand()) / RAND_MAX * 0.1 - 0.05; // Small random angle
        theta_dot = 0.0;
    }

    std::tuple<torch::Tensor, float, bool> step(float action) {
        action = std::max(-max_torque, std::min(max_torque, action));
        float theta_ddot = -g / l * std::sin(theta) + action / (m * l * l);
        theta_dot += theta_ddot * dt;
        theta_dot = std::max(-max_theta_dot, std::min(max_theta_dot, theta_dot));
        theta += theta_dot * dt;
        theta = std::fmod(theta + M_PI, 2 * M_PI) - M_PI; // Keep theta in [-pi, pi]

        float reward = -(theta * theta + 0.1 * theta_dot * theta_dot + 0.001 * action * action);
        bool done = (std::abs(theta) > max_theta);
        return {get_state(), reward, done};
    }
};

// Replay Buffer
struct ReplayBuffer {
    std::vector<torch::Tensor> states, actions, rewards, next_states;
    std::vector<bool> dones;
    size_t max_size = 100000;
    size_t size = 0;
    size_t idx = 0;

    void add(torch::Tensor state, torch::Tensor action, float reward, torch::Tensor next_state, bool done) {
        if (states.size() < max_size) {
            states.push_back(state.clone());
            actions.push_back(action.clone());
            rewards.push_back(torch::tensor({reward}, torch::kFloat32));
            next_states.push_back(next_state.clone());
            dones.push_back(done);
        } else {
            states[idx] = state.clone();
            actions[idx] = action.clone();
            rewards[idx] = torch::tensor({reward}, torch::kFloat32);
            next_states[idx] = next_state.clone();
            dones[idx] = done;
        }
        idx = (idx + 1) % max_size;
        size = std::min(size + 1, max_size);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(size_t batch_size) {
        std::vector<size_t> indices(batch_size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, size - 1);
        for (size_t i = 0; i < batch_size; ++i) {
            indices[i] = dis(gen);
        }

        std::vector<torch::Tensor> sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones;
        for (auto i : indices) {
            sampled_states.push_back(states[i]);
            sampled_actions.push_back(actions[i]);
            sampled_rewards.push_back(rewards[i]);
            sampled_next_states.push_back(next_states[i]);
            sampled_dones.push_back(torch::tensor({dones[i] ? 1.0f : 0.0f}, torch::kFloat32));
        }

        return {
                torch::stack(sampled_states),
                torch::stack(sampled_actions),
                torch::stack(sampled_rewards),
                torch::stack(sampled_next_states),
                torch::stack(sampled_dones)
        };
    }
};

// Actor Network
struct Actor : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    Actor() {
        fc1 = register_module("fc1", torch::nn::Linear(2, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::tanh(fc3->forward(x)) * 2.0; // Scale to [-2, 2]
        return x;
    }
};

// Critic Network
struct Critic : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    Critic() {
        fc1 = register_module("fc1", torch::nn::Linear(3, 128)); // State (2) + Action (1)
        fc2 = register_module("fc2", torch::nn::Linear(128, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 1));
    }

    torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        auto x = torch::cat({state, action}, 1);
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};

// Ornstein-Uhlenbeck Noise for exploration
struct OUNoise {
    float mu, theta, sigma;
    torch::Tensor state;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution;

    OUNoise(float mu = 0.0, float theta = 0.15, float sigma = 0.2)
            : mu(mu), theta(theta), sigma(sigma), state(torch::zeros({1})), distribution(0.0, 1.0) {
        std::random_device rd;
        generator.seed(rd());
    }

    void reset() {
        state = torch::zeros({1});
    }

    torch::Tensor sample() {
        auto x = state.item<float>();
        auto dx = theta * (mu - x) + sigma * distribution(generator);
        state = torch::tensor({x + dx}, torch::kFloat32);
        return state;
    }
};

// DDPG Agent
struct DDPG {
    std::shared_ptr<Actor> actor, actor_target;
    std::shared_ptr<Critic> critic, critic_target;
    torch::optim::Adam actor_optimizer, critic_optimizer;
    ReplayBuffer buffer;
    OUNoise noise;
    float gamma = 0.99;
    float tau = 0.005;

    DDPG()
            : actor(std::make_shared<Actor>()),
              actor_target(std::make_shared<Actor>()),
              critic(std::make_shared<Critic>()),
              critic_target(std::make_shared<Critic>()),
              actor_optimizer(actor->parameters(), torch::optim::AdamOptions(1e-4)),
              critic_optimizer(critic->parameters(), torch::optim::AdamOptions(1e-3)) {
        actor_target->load_state_dict(actor->state_dict());
        critic_target->load_state_dict(critic->state_dict());
    }

    void update(torch::Tensor state, torch::Tensor action, float reward, torch::Tensor next_state, bool done) {
        buffer.add(state, action, reward, next_state, done);
        if (buffer.size < 1000) return;

        auto [states, actions, rewards, next_states, dones] = buffer.sample(64);

        // Update Critic
        critic->zero_grad();
        auto target_actions = actor_target->forward(next_states);
        auto target_q = critic_target->forward(next_states, target_actions);
        auto y = rewards + gamma * target_q * (1 - dones);
        auto q = critic->forward(states, actions);
        auto critic_loss = torch::mse_loss(q, y.detach());
        critic_loss.backward();
        critic_optimizer.step();

        // Update Actor
        actor->zero_grad();
        auto actor_loss = -critic->forward(states, actor->forward(states)).mean();
        actor_loss.backward();
        actor_optimizer.step();

        // Soft update target networks
        for (size_t i = 0; i < actor->parameters().size(); ++i) {
            auto p = actor->parameters()[i];
            auto pt = actor_target->parameters()[i];
            pt.set_data(tau * p + (1 - tau) * pt);
        }
        for (size_t i = 0; i < critic->parameters().size(); ++i) {
            auto p = critic->parameters()[i];
            auto pt = critic_target->parameters()[i];
            pt.set_data(tau * p + (1 - tau) * pt);
        }
    }

    torch::Tensor select_action(torch::Tensor state, bool explore = true) {
        auto action = actor->forward(state);
        if (explore) {
            action += noise.sample();
            action = torch::clamp(action, -2.0, 2.0);
        }
        return action;
    }
};

int main() {
    torch::manual_seed(0);
    srand(static_cast<unsigned>(time(0)));

    PendulumEnv env;
    DDPG agent;
    const int episodes = 1000;
    const int max_steps = 200;

    for (int episode = 0; episode < episodes; ++episode) {
        env.reset();
        auto state = env.get_state();
        float total_reward = 0.0;

        for (int step = 0; step < max_steps; ++step) {
            auto action = agent.select_action(state);
            auto [next_state, reward, done] = env.step(action.item<float>());
            agent.update(state, action, reward, next_state, done);
            state = next_state;
            total_reward += reward;

            if (done) break;
        }

        if (episode % 100 == 0) {
            std by::cout << "Episode: " << episode << ", Total Reward: " << total_reward << std::endl;
        }
    }

    return 0;
}

namespace xt::models
{
    DDPG::DDPG(int num_classes, int in_channels)
    {
    }

    DDPG::DDPG(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DDPG::reset()
    {
    }

    auto DDPG::forward(std::initializer_list<std::any> tensors) -> std::any
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
