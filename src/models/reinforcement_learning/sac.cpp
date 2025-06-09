#include "include/models/reinforcement_learning/sac.h"


using namespace std;


#include <torch/torch.h>
#include <vector>
#include <random>
#include <iostream>
#include <deque>
#include <cmath>

// Simulated CartPole Environment
struct CartPoleEnv {
    std::vector<float> state;
    float gravity = 9.8, masscart = 1.0, masspole = 0.1, length = 0.5;
    float tau = 0.02, force_mag = 10.0;
    bool done = false;
    int steps = 0;
    const int max_steps = 200;
    std::mt19937 gen;

    CartPoleEnv() : state(4, 0.0), gen(std::random_device{}()) {
        reset();
    }

    void reset() {
        std::uniform_real_distribution<float> dist(-0.05, 0.05);
        state = {dist(gen), dist(gen), dist(gen), dist(gen)};
        done = false;
        steps = 0;
    }

    std::tuple<std::vector<float>, float, bool> step(int action) {
        float x = state[0], x_dot = state[1], theta = state[2], theta_dot = state[3];
        float force = (action == 1) ? force_mag : -force_mag;
        float costheta = std::cos(theta), sintheta = std::sin(theta);
        float temp = (force + masspole * length * theta_dot * theta_dot * sintheta) / (masscart + masspole);
        float theta_acc = (gravity * sintheta - costheta * temp) /
                          (length * (4.0 / 3.0 - masspole * costheta * costheta / (masscart + masspole)));
        float x_acc = temp - masspole * length * theta_acc * costheta / (masscart + masspole);

        x += tau * x_dot;
        x_dot += tau * x_acc;
        theta += tau * theta_dot;
        theta_dot += tau * theta_acc;

        state = {x, x_dot, theta, theta_dot};
        steps++;

        float reward = 1.0;
        done = (std::abs(x) > 2.4 || std::abs(theta) > 12.0 * M_PI / 180.0 || steps >= max_steps);
        if (done && steps < max_steps) {
            reward = -10.0;
        }

        return {state, reward, done};
    }
};

// Replay Buffer
struct ReplayBuffer {
    struct Transition {
        std::vector<float> state, next_state;
        int action;
        float reward;
        bool done;
    };

    std::deque<Transition> buffer;
    size_t capacity;
    std::mt19937 gen;

    ReplayBuffer(size_t capacity_) : capacity(capacity_), gen(std::random_device{}()) {}

    void push(const std::vector<float>& state, int action, float reward,
              const std::vector<float>& next_state, bool done) {
        if (buffer.size() >= capacity) {
            buffer.pop_front();
        }
        buffer.push_back({state, next_state, action, reward, done});
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(size_t batch_size) {
        std::uniform_int_distribution<size_t> dist(0, buffer.size() - 1);
        std::vector<float> states, next_states, rewards, dones;
        std::vector<int64_t> actions;

        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = dist(gen);
            auto& t = buffer[idx];
            states.insert(states.end(), t.state.begin(), t.state.end());
            next_states.insert(next_states.end(), t.next_state.begin(), t.next_state.end());
            actions.push_back(t.action);
            rewards.push_back(t.reward);
            dones.push_back(static_cast<float>(t.done));
        }

        auto state_tensor = torch::tensor(states).view({static_cast<int64_t>(batch_size), 4});
        auto action_tensor = torch::tensor(actions).view({static_cast<int64_t>(batch_size), 1});
        auto reward_tensor = torch::tensor(rewards).view({static_cast<int64_t>(batch_size), 1});
        auto next_state_tensor = torch::tensor(next_states).view({static_cast<int64_t>(batch_size), 4});
        auto done_tensor = torch::tensor(dones).view({static_cast<int64_t>(batch_size), 1});

        return {state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor};
    }

    size_t size() const { return buffer.size(); }
};

// Actor Network
struct ActorNetwork : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    ActorNetwork(int state_dim, int action_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, action_dim));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        auto logits = fc2->forward(x);
        auto probs = torch::softmax(logits, -1);
        auto dist = torch::distributions::Categorical(probs);
        auto action = dist.sample();
        auto log_prob = dist.log_prob(action);
        return {action, log_prob};
    }

    torch::Tensor get_log_prob(torch::Tensor x, torch::Tensor action) {
        x = torch::relu(fc1->forward(x));
        auto logits = fc2->forward(x);
        auto probs = torch::softmax(logits, -1);
        auto dist = torch::distributions::Categorical(probs);
        return dist.log_prob(action.squeeze(-1));
    }
};

// Q-Network
struct QNetwork : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    QNetwork(int state_dim, int action_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, action_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        return fc2->forward(x);
    }
};

// SAC Agent
class SACAgent {
public:
    ActorNetwork actor_net;
    QNetwork q1_net, q2_net, q1_target_net, q2_target_net;
    ReplayBuffer replay_buffer;
    torch::optim::Adam actor_optimizer, q_optimizer;
    torch::Device device;
    torch::Tensor log_alpha;
    torch::optim::Adam alpha_optimizer;
    float gamma, tau, target_entropy, alpha_lr;
    int state_dim, action_dim, batch_size, update_freq, step_count;

    SACAgent(int state_dim_, int action_dim_, float gamma_, float tau_, float lr_, float alpha_lr_,
             size_t buffer_capacity_, int batch_size_, int update_freq_)
        : actor_net(state_dim_, action_dim_),
          q1_net(state_dim_, action_dim_),
          q2_net(state_dim_, action_dim_),
          q1_target_net(state_dim_, action_dim_),
          q2_target_net(state_dim_, action_dim_),
          replay_buffer(buffer_capacity_),
          actor_optimizer(actor_net.parameters(), torch::optim::AdamOptions(lr_)),
          q_optimizer({q1_net.parameters(), q2_net.parameters()}, torch::optim::AdamOptions(lr_)),
          device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          log_alpha(torch::tensor(0.0, torch::requires_grad(true).to(device))),
          alpha_optimizer({log_alpha}, torch::optim::AdamOptions(alpha_lr_)),
          gamma(gamma_),
          tau(tau_),
          target_entropy(-std::log(1.0f / action_dim_)),
          alpha_lr(alpha_lr_),
          state_dim(state_dim_),
          action_dim(action_dim_),
          batch_size(batch_size_),
          update_freq(update_freq_),
          step_count(0) {
        actor_net.to(device);
        q1_net.to(device);
        q2_net.to(device);
        q1_target_net.to(device);
        q2_target_net.to(device);
        q1_target_net.load_state_dict(q1_net.state_dict());
        q2_target_net.load_state_dict(q2_net.state_dict());
        q1_target_net.eval();
        q2_target_net.eval();
    }

    int select_action(torch::Tensor state) {
        actor_net.eval();
        torch::NoGradGuard no_grad;
        state = state.to(device);
        auto [action, _] = actor_net.forward(state);
        actor_net.train();
        return action.cpu().item<int64_t>();
    }

    void update_target_networks() {
        for (auto& param : q1_target_net.named_parameters()) {
            auto target_param = q1_target_net.named_parameters()[param.key()];
            target_param.value().copy_(tau * param.value() + (1 - tau) * target_param.value());
        }
        for (auto& param : q2_target_net.named_parameters()) {
            auto target_param = q2_target_net.named_parameters()[param.key()];
            target_param.value().copy_(tau * param.value() + (1 - tau) * target_param.value());
        }
    }

    void train_step() {
        if (replay_buffer.size() < batch_size) return;

        auto [states, actions, rewards, next_states, dones] = replay_buffer.sample(batch_size);
        states = states.to(device);
        actions = actions.to(device);
        rewards = rewards.to(device);
        next_states = next_states.to(device);
        dones = dones.to(device);

        float alpha = torch::exp(log_alpha).item<float>();

        // Q-Network Update
        auto q1_values = q1_net.forward(states).gather(1, actions);
        auto q2_values = q2_net.forward(states).gather(1, actions);

        torch::NoGradGuard no_grad;
        auto [next_actions, next_log_probs] = actor_net.forward(next_states);
        auto next_q1 = q1_target_net.forward(next_states);
        auto next_q2 = q2_target_net.forward(next_states);
        auto next_q = torch::min(next_q1, next_q2);
        auto next_v = (next_q - alpha * next_log_probs.unsqueeze(-1)).sum(-1, true);
        auto target_q = rewards + gamma * (1 - dones) * next_v;

        auto q1_loss = torch::mse_loss(q1_values, target_q);
        auto q2_loss = torch::mse_loss(q2_values, target_q);
        auto q_loss = (q1_loss + q2_loss) / 2;

        q_optimizer.zero_grad();
        q_loss.backward();
        torch::nn::utils::clip_grad_norm_({q1_net.parameters(), q2_net.parameters()}, 1.0);
        q_optimizer.step();

        // Actor Update
        auto [actions_pred, log_probs_pred] = actor_net.forward(states);
        auto q1_pred = q1_net.forward(states);
        auto q2_pred = q2_net.forward(states);
        auto q_pred = torch::min(q1_pred, q2_pred);
        auto actor_loss = (alpha * log_probs_pred.unsqueeze(-1) - q_pred).mean();

        actor_optimizer.zero_grad();
        actor_loss.backward();
        torch::nn::utils::clip_grad_norm_(actor_net.parameters(), 1.0);
        actor_optimizer.step();

        // Alpha Update
        auto alpha_loss = -(log_alpha * (log_probs_pred + target_entropy).detach()).mean();

        alpha_optimizer.zero_grad();
        alpha_loss.backward();
        alpha_optimizer.step();

        // Update target networks
        update_target_networks();
    }

    void collect_trajectory(CartPoleEnv& env, int max_steps) {
        env.reset();
        auto state = torch::tensor(env.state).view({1, state_dim});
        float episode_reward = 0.0;
        int steps = 0;

        while (!env.done && steps < max_steps) {
            int action = select_action(state);
            auto [next_state, reward, done] = env.step(action);
            auto next_state_tensor = torch::tensor(next_state).view({1, state_dim});
            replay_buffer.push(state.view(-1).tolist<float>(), action, reward,
                               next_state_tensor.view(-1).tolist<float>(), done);
            episode_reward += reward;
            steps++;

            if (replay_buffer.size() >= batch_size && step_count % update_freq == 0) {
                train_step();
            }

            state = next_state_tensor;
            step_count++;
        }

        return {episode_reward, steps};
    }
};

int main() {
    torch::manual_seed(42);
    std::mt19937 gen(std::random_device{}());

    // Environment and agent
    CartPoleEnv env;
    SACAgent agent(
        4, 2, 0.99, 0.005, 0.001, 0.001, 10000, 64, 4 // state_dim, action_dim, gamma, tau, lr, alpha_lr, buffer_capacity, batch_size, update_freq
    );

    // Training loop
    const int episodes = 500;
    const int max_steps = 200;

    for (int episode = 0; episode < episodes; ++episode) {
        auto [episode_reward, steps] = agent.collect_trajectory(env, max_steps);

        if (episode % 10 == 0) {
            std::cout << "Episode: " << episode << ", Reward: " << episode_reward
                      << ", Steps: " << steps << ", Alpha: " << torch::exp(agent.log_alpha).item<float>()
                      << std::endl;
        }
    }

    // Save networks
    torch::save(agent.actor_net, "sac_actor.pt");
    torch::save(agent.q1_net, "sac_q1.pt");
    torch::save(agent.q2_net, "sac_q2.pt");

    return 0;
}


namespace xt::models
{
    SAC::SAC(int num_classes, int in_channels)
    {
    }

    SAC::SAC(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void SAC::reset()
    {
    }

    auto SAC::forward(std::initializer_list<std::any> tensors) -> std::any
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
