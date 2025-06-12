#include "include/losses/wgan_gp_loss.h" // Your header
#include <torch/torch.h> // For torch functionalities
#include <string>        // For std::string
#include <memory>        // For std::shared_ptr
#include <any>           // For std::any
#include <vector>        // For std::vector (used by autograd::grad)

// Assuming xt::Module is defined something like this (typically in another header):

namespace xt::losses
{
    torch::Tensor wgan_gp_loss(const torch::Tensor& d_real_logits, const torch::Tensor& d_fake_logits,
                               const torch::Tensor& real_data, const torch::Tensor& fake_data,
                               // Corrected type: pass by const reference to avoid shared_ptr copy
                               const std::shared_ptr<xt::Module>& discriminator,
                               const std::string& mode,
                               float lambda_gp = 10.0f, // Use 'f' suffix for float literals
                               float eps = 1e-6f)
    {
        // Ensure inputs are on the same device
        auto device = d_real_logits.device();
        TORCH_CHECK(d_fake_logits.device() == device, "Real and fake logits must be on the same device");
        TORCH_CHECK(real_data.device() == device, "Real data must be on the same device");
        TORCH_CHECK(fake_data.device() == device, "Fake data must be on the same device");

        // Input validation
        TORCH_CHECK(discriminator, "Discriminator module cannot be null."); // Null check
        TORCH_CHECK(d_real_logits.dim() == 1, "Real logits must be 1D (batch_size)");
        TORCH_CHECK(d_fake_logits.dim() == 1, "Fake logits must be 1D (batch_size)");
        TORCH_CHECK(real_data.sizes() == fake_data.sizes(), "Real and fake data must have the same shape");
        TORCH_CHECK(d_real_logits.size(0) == d_fake_logits.size(0) && d_real_logits.size(0) == real_data.size(0),
                    "Batch sizes must match");
        TORCH_CHECK(mode == "discriminator" || mode == "generator", "Mode must be 'discriminator' or 'generator'");
        TORCH_CHECK(lambda_gp >= 0.0f, "Gradient penalty weight must be non-negative");

        // Generator loss: -mean(D(fake))
        if (mode == "generator")
        {
            return -d_fake_logits.mean();
        }

        // Discriminator loss: mean(D(fake)) - mean(D(real)) + lambda_gp * gradient_penalty
        torch::Tensor base_loss = d_fake_logits.mean() - d_real_logits.mean();

        // Compute gradient penalty
        int64_t batch_size = real_data.size(0); // Use int64_t for sizes
        // Generate interpolation coefficients
        // Ensure options match data: device, dtype (implicitly float from real_data)
        auto alpha_options = torch::TensorOptions().device(device).dtype(real_data.dtype());
        torch::Tensor alpha = torch::rand({batch_size, 1, 1, 1}, alpha_options);
        // No need to expand_as if subsequent ops broadcast correctly, but expand_as is safer.
        alpha = alpha.expand_as(real_data);

        // Interpolate between real and fake data
        // .detach() is used on real_data and fake_data to avoid backpropagating GP through G or data loading.
        torch::Tensor interpolates = alpha * real_data.detach() + (1.0f - alpha) * fake_data.detach();
        interpolates.set_requires_grad(true);

        // Compute discriminator output for interpolated samples
        std::any d_interpolates_any;
        try
        {
            // The discriminator's forward method is called here.
            // It's crucial that this forward method is implemented correctly
            // in the concrete discriminator class derived from xt::Module.
            d_interpolates_any = discriminator->forward({interpolates});
        }
        catch (const std::exception& e)
        {
            TORCH_CHECK(false, "Exception during discriminator->forward(): ", e.what());
        }

        torch::Tensor d_interpolates;
        try
        {
            d_interpolates = std::any_cast<torch::Tensor>(d_interpolates_any);
        }
        catch (const std::bad_any_cast& e)
        {
            TORCH_CHECK(false, "Discriminator forward() did not return a torch::Tensor. Error: ", e.what(),
                        ". Actual type: ", d_interpolates_any.type().name());
        }

        TORCH_CHECK(d_interpolates.dim() == 1 && d_interpolates.size(0) == batch_size,
                    "Discriminator output for interpolates must be 1D (batch_size)");
        TORCH_CHECK(d_interpolates.device() == device,
                    "Discriminator output for interpolates must be on the same device");


        // Compute gradients of discriminator output w.r.t. interpolated inputs
        // grad_outputs should be on the same device as d_interpolates
        torch::Tensor grad_outputs = torch::ones_like(d_interpolates, torch::requires_grad(false));

        std::vector<torch::Tensor> grads;
        try
        {
            grads = torch::autograd::grad(
                /*outputs=*/{d_interpolates},
                            /*inputs=*/{interpolates},
                            /*grad_outputs=*/{grad_outputs},
                            /*retain_graph=*/true, // Needed if D loss will be backpropped further through these ops
                            /*create_graph=*/true, // Needed to compute gradients of gradients (for GP)
                            /*allow_unused=*/false // Inputs to grad must have requires_grad=true
            );
        }
        catch (const std::exception& e)
        {
            TORCH_CHECK(false, "Exception during torch::autograd::grad for gradient penalty: ", e.what());
        }

        TORCH_CHECK(!grads.empty() && grads[0].defined(), "Gradient computation failed or returned undefined tensor.");
        torch::Tensor gradients = grads[0];

        // Compute gradient norm
        gradients = gradients.view({batch_size, -1});
        // Add eps to norm before squaring to prevent nan/inf if norm is zero and power is < 1 (not an issue for pow(x,2))
        // or if norm is very small, leading to issues if 1.0 is subtracted.
        torch::Tensor gradient_norm = gradients.norm(2, /*dim=*/1, /*keepdim=*/false); // Norm along feature dimension
        torch::Tensor gradient_penalty = lambda_gp * torch::pow(gradient_norm - 1.0f, 2).mean();

        // Combine base loss and gradient penalty
        return base_loss + gradient_penalty;
    }

    // Example implementation for WGANGPLoss::forward
    // This assumes WGANGPLoss is an xt::Module itself and its constructor
    // might take lambda_gp. The discriminator would typically be a member
    // of the WGANGPLoss class, or passed in if the design dictates.
    auto WGANGPLoss::forward(std::initializer_list<std::any> inputs) -> std::any
    {
        // This is a more typical setup for a loss module where some parameters are members.
        // For this example, we'll assume they are all passed in `inputs`.
        // Order: d_real_logits, d_fake_logits, real_data, fake_data, discriminator_ptr, mode_str, [lambda_gp], [eps]
        TORCH_CHECK(inputs.size() >= 6,
                    "WGANGPLoss::forward expects at least 6 arguments: "
                    "d_real_logits (Tensor), d_fake_logits (Tensor), real_data (Tensor), fake_data (Tensor), "
                    "discriminator (std::shared_ptr<xt::Module>), mode (std::string)");

        auto it = inputs.begin();
        const auto& d_real_logits = std::any_cast<const torch::Tensor&>(*it++);
        const auto& d_fake_logits = std::any_cast<const torch::Tensor&>(*it++);
        const auto& real_data = std::any_cast<const torch::Tensor&>(*it++);
        const auto& fake_data = std::any_cast<const torch::Tensor&>(*it++);
        // Assuming the std::any holds a const std::shared_ptr<xt::Module>& or std::shared_ptr<xt::Module>
        const auto& discriminator_ptr = std::any_cast<const std::shared_ptr<xt::Module>&>(*it++);
        const auto& mode_str = std::any_cast<const std::string&>(*it++);

        float lambda_gp_val = 10.0f; // Default, or from member this->lambda_gp_
        if (inputs.size() > 6)
        {
            lambda_gp_val = std::any_cast<float>(*it++);
        }

        float eps_val = 1e-6f; // Default, or from member this->eps_
        if (inputs.size() > 7)
        {
            eps_val = std::any_cast<float>(*it++);
        }

        return xt::losses::wgan_gp_loss(d_real_logits, d_fake_logits,
                                        real_data, fake_data,
                                        discriminator_ptr, mode_str,
                                        lambda_gp_val, eps_val);
    }
} // namespace xt::losses
