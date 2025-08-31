# Reinforcement Learning Models

Reinforcement Learning (RL) is a paradigm of machine learning where an "agent" learns to make decisions by performing actions in an "environment" to maximize a cumulative reward.

Unlike supervised learning, RL models are not trained on a fixed dataset. Instead, they are policies and/or value functions that an agent uses to interact with an environment and learn from the feedback it receives.

xTorch provides implementations of several major RL algorithms, encapsulating the underlying neural network architectures (the policies and value functions) that power them.

All RL models are located under the `xt::models` namespace and their headers can be found in the `<xtorch/models/reinforcement_learning/>` directory.

## General Usage

RL models are used differently from standard supervised models. Instead of a single `forward` pass on a batch of data, they are typically used within an "agent-environment loop."

-   A **Policy Network** takes the current state (observation) from the environment and outputs a probability distribution over possible actions.
-   A **Value Network** (or Q-Network) takes the current state and outputs an estimated value for each possible action (the expected future reward).

The examples below show how to instantiate and use these two core components.

### Example: Using a Q-Network (for DQN)

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // --- Environment & Model Properties ---
    const int num_observations = 4; // E.g., state space size for CartPole
    const int num_actions = 2;      // E.g., number of possible actions

    // --- Instantiate a DQN Model ---
    // This is the Q-Network that approximates the action-value function.
    xt::models::DQN model(num_observations, num_actions);
    model.to(device);
    model.eval();

    std::cout << "DQN Model (Q-Network) Instantiated." << std::endl;

    // --- Get Action Values for a Given State ---
    // Create a dummy observation from the environment
    auto state = torch::randn({1, num_observations}).to(device); // Batch size of 1

    // The model's forward pass returns the Q-value for each action
    auto action_values = model.forward(state);

    // The agent would then use an epsilon-greedy strategy to select an action
    auto best_action = torch::argmax(action_values, /*dim=*/1);

    std::cout << "Action values: " << action_values << std::endl;
    std::cout << "Best action: " << best_action.item<long>() << std::endl;
}
```

### Example: Using a Policy Network (for Actor-Critic)

```cpp
// --- Instantiate an Actor-Critic Model (like A3C) ---
// Note: A3C often has a shared body with two heads (policy and value).
// xt::models::A3C model(num_observations, num_actions);
// model.to(device);
// auto [policy_logits, value_estimate] = model.forward(state);

// The policy head gives logits, which are converted to a probability distribution
// auto action_probabilities = torch::softmax(policy_logits, /*dim=*/-1);
// torch::distributions::Categorical dist(action_probabilities);
// auto action = dist.sample(); // Sample an action from the policy
```

---

## Available Models by Family

### Value-Based Methods

These methods learn a value function that estimates the expected return for taking an action in a given state. The policy is often implicit (e.g., "always take the action with the highest value").

| Model | Description | Header File |
|---|---|---|
| `DQN` | **Deep Q-Network**. A foundational algorithm that uses a deep neural network to approximate the optimal action-value function, Q*. | `dqn.h` |
| `DoubleDQN`| An improvement over DQN that decouples action selection from action evaluation to reduce overestimation of Q-values. | `double_dqn.h` |
| `DuelingDQN`| An architecture that separates the estimation of state values and action advantages, leading to better policy evaluation. | `dueling_dqn.h` |
| `Rainbow` | A combination of seven improvements to DQN (including Double, Dueling, Prioritized Replay, etc.) into a single, high-performing agent. | `rainbow.h` |

### Policy-Based & Actor-Critic Methods

These methods directly learn a policy that maps states to actions. Actor-Critic methods learn both a policy (the actor) and a value function (the critic) simultaneously.

| Model | Description | Header File |
|---|---|---|
| `A3C` | **Asynchronous Advantage Actor-Critic**. A classic parallel RL algorithm. | `a3c.h` |
| `PPO` | **Proximal Policy Optimization**. A highly effective and stable actor-critic method, often a default choice for many continuous control problems. | `pro.h` |
| `DDPG`| **Deep Deterministic Policy Gradient**. An actor-critic, model-free algorithm for continuous action spaces. | `ddpg.h` |
| `TD3` | **Twin Delayed DDPG**. An improvement over DDPG that addresses Q-value overestimation by using two critic networks. | `td3.h` |
| `SAC` | **Soft Actor-Critic**. An off-policy actor-critic algorithm based on the maximum entropy framework, known for its sample efficiency and stability. | `sac.h` |

### Model-Based & Planning Methods

These methods learn a model of the environment and use it to plan future actions.

| Model | Description | Header File |
|---|---|---|
| `AlphaGo` | The pioneering deep RL program that defeated the world champion Go player, combining Monte Carlo tree search with deep neural networks. | `alpha_go.h` |
| `AlphaZero` | A more generalized and powerful version of AlphaGo that learns entirely from self-play and mastered Go, chess, and shogi. | `alpha_zero.h` |
| `MuZero` | A powerful successor to AlphaZero that achieves superhuman performance by learning a model of the environment and applying tree-based search. | `mu_zero.h` |

### Other Architectures
| Model | Description | Header File |
|---|---|---|
| `IMPALA`| **Importance Weighted Actor-Learner Architecture**. A scalable, distributed agent that can be used for both single and multi-task reinforcement learning. | `impala.h` |
