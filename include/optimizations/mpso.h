#pragma once


#include "common.h"

namespace xt
{
    class Module;
}

// Represents a single particle in the swarm
struct Particle {
    std::vector<torch::Tensor> position; // The model weights
    std::vector<torch::Tensor> velocity;
    std::vector<torch::Tensor> pbest_position; // Personal best position
    double pbest_fitness = std::numeric_limits<double>::infinity();

    Particle() = default;
    // Constructor to initialize a particle
    Particle(const std::vector<torch::Tensor>& initial_position);
    // Deep clone method
    Particle clone() const;
};

// --- Options for MPSO ---
struct MPSOOptions {
    int population_size = 20;
    // PSO hyperparameters
    double inertia_weight = 0.5;
    double cognitive_weight = 1.5; // c1: Pull towards personal best
    double social_weight = 1.5;    // c2: Pull towards global best
    double gradient_weight = 1.0;  // c3: Pull in direction of gradient (hybrid part)
};

// --- MPSO Optimizer Class (Meta-Optimizer) ---
class MPSO {
public:
    MPSO(std::shared_ptr<xt::Module> model, MPSOOptions options);

    // The step function needs data and a loss function to evaluate fitness
    double step(const torch::Tensor& input, const torch::Tensor& target,
                const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& loss_fn);

    // Get the current best model parameters to use for inference
    std::vector<torch::Tensor> get_best_params() const;

private:
    std::shared_ptr<xt::Module> model_; // The model whose structure we use
    MPSOOptions options_;

    // The swarm of particles
    std::vector<Particle> swarm_;

    // Global best state
    std::vector<torch::Tensor> gbest_position_;
    double gbest_fitness_ = std::numeric_limits<double>::infinity();

    void _update_particle(Particle& particle, const torch::Tensor& input, const torch::Tensor& target,
                          const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& loss_fn);
};

