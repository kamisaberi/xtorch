#include "include/optimizations/mpso.h"
#include <stdexcept>

#include "base/module.h"

namespace xt::optim
{
    // --- Particle Methods ---
    Particle::Particle(const std::vector<torch::Tensor>& initial_position)
    {
        torch::NoGradGuard no_grad;
        for (const auto& p : initial_position)
        {
            position.push_back(p.detach().clone());
            velocity.push_back(torch::zeros_like(p));
            pbest_position.push_back(p.detach().clone());
        }
    }

    Particle Particle::clone() const
    {
        Particle cloned_particle;
        torch::NoGradGuard no_grad;
        for (const auto& p : position) cloned_particle.position.push_back(p.detach().clone());
        for (const auto& v : velocity) cloned_particle.velocity.push_back(v.detach().clone());
        for (const auto& pb : pbest_position) cloned_particle.pbest_position.push_back(pb.detach().clone());
        cloned_particle.pbest_fitness = pbest_fitness;
        return cloned_particle;
    }


    // --- MPSO Implementation ---
    MPSO::MPSO(std::shared_ptr<xt::Module> model, MPSOOptions options)
        : model_(model), options_(options)
    {
        TORCH_CHECK(model_ != nullptr, "A valid model must be provided.");

        // Initialize the swarm
        auto initial_params = model_->parameters();
        for (int i = 0; i < options_.population_size; ++i)
        {
            // Create the first particle from the initial model weights
            Particle p(initial_params);
            if (i > 0)
            {
                // Jitter subsequent particles slightly to explore different starting points
                for (auto& param_tensor : p.position)
                {
                    param_tensor.add_(torch::randn_like(param_tensor) * 0.01);
                }
            }
            swarm_.push_back(p);
        }

        // Initialize global best from the first particle
        gbest_position_ = swarm_[0].position;
        // gbest_fitness will be updated on the first step
    }

    double MPSO::step(const torch::Tensor& input, const torch::Tensor& target,
                      const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& loss_fn)
    {
        // Iterate through each particle and update it
        for (auto& particle : swarm_)
        {
            _update_particle(particle, input, target, loss_fn);
        }

        return gbest_fitness_;
    }

    // Private helper to update a single particle
    void MPSO::_update_particle(Particle& particle, const torch::Tensor& input, const torch::Tensor& target,
                                const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& loss_fn)
    {
        torch::NoGradGuard no_grad_guard; // Most operations are no_grad

        // Temporarily load the particle's weights into the model to evaluate it
        auto original_params = model_->parameters();
        for (size_t i = 0; i < original_params.size(); ++i)
        {
            original_params[i].data().copy_(particle.position[i]);
        }

        // --- Evaluate Fitness and Compute Gradient ---
        double current_fitness;
        {
            // We need gradients enabled just for this block
            torch::GradMode::set_enabled(true);
            model_->zero_grad();
            auto output = std::any_cast<torch::Tensor>(model_->forward({input}));
            auto loss = loss_fn(output, target);
            current_fitness = loss.item<double>();
            loss.backward();
        } // Gradients disabled again by NoGradGuard destructor

        // --- Update Personal Best (pbest) ---
        if (current_fitness < particle.pbest_fitness)
        {
            particle.pbest_fitness = current_fitness;
            for (size_t i = 0; i < particle.position.size(); ++i)
            {
                particle.pbest_position[i].copy_(particle.position[i]);
            }
        }

        // --- Update Global Best (gbest) ---
        // This part should be synchronized in a real parallel implementation
        if (current_fitness < gbest_fitness_)
        {
            gbest_fitness_ = current_fitness;
            for (size_t i = 0; i < particle.position.size(); ++i)
            {
                gbest_position_[i].copy_(particle.position[i]);
            }
        }

        // --- Update Velocity and Position for each parameter tensor ---
        for (size_t i = 0; i < particle.position.size(); ++i)
        {
            auto& vel = particle.velocity[i];
            auto& pos = particle.position[i];
            auto& pbest = particle.pbest_position[i];
            auto& gbest = gbest_position_[i];

            // Random numbers for stochastic component
            auto r1 = torch::rand_like(pos);
            auto r2 = torch::rand_like(pos);
            auto r3 = torch::rand_like(pos);

            // Velocity update components
            auto inertia_term = options_.inertia_weight * vel;
            auto cognitive_term = options_.cognitive_weight * r1 * (pbest - pos);
            auto social_term = options_.social_weight * r2 * (gbest - pos);

            // Hybrid part: Use the gradient as another force
            auto& grad = model_->parameters()[i].grad();
            auto gradient_term = options_.gradient_weight * r3 * (-grad);

            // Update velocity
            vel = inertia_term + cognitive_term + social_term + gradient_term;

            // Update position
            pos.add_(vel);
        }
    }

    std::vector<torch::Tensor> MPSO::get_best_params() const
    {
        std::vector<torch::Tensor> best_params_cloned;
        for (const auto& p : gbest_position_)
        {
            best_params_cloned.push_back(p.detach().clone());
        }
        return best_params_cloned;
    }
}