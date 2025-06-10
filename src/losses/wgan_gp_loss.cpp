#include "include/losses/wgan_gp_loss.h"

namespace xt::losses
{
    /**
     * Wasserstein GAN with Gradient Penalty (WGAN-GP) Loss function.
     * Computes discriminator or generator loss with gradient penalty for 1-Lipschitz enforcement.
     * @param d_real_logits Discriminator logits for real samples, shape: [batch_size]
     * @param d_fake_logits Discriminator logits for fake samples, shape: [batch_size]
     * @param real_data Real input data for gradient penalty, shape: [batch_size, channels, height, width]
     * @param fake_data Fake input data for gradient penalty, shape: [batch_size, channels, height, width]
     * @param discriminator Discriminator module to compute gradients
     * @param mode Specifies the loss type: "discriminator" or "generator"
     * @param lambda_gp Gradient penalty weight, default: 10.0
     * @param eps Small value for numerical stability, default: 1e-6
     * @return Scalar tensor containing the WGAN-GP Loss
     */
    torch::Tensor wgan_gp_loss(const torch::Tensor& d_real_logits, const torch::Tensor& d_fake_logits,
                               const torch::Tensor& real_data, const torch::Tensor& fake_data,
                               std::shared_ptr<xt::Module&> discriminator, const std::string& mode,
                               float lambda_gp = 10.0,
                               float eps = 1e-6)
    {
        // Ensure inputs are on the same device
        auto device = d_real_logits.device();
        TORCH_CHECK(d_fake_logits.device() == device, "Real and fake logits must be on the same device");
        TORCH_CHECK(real_data.device() == device, "Real data must be on the same device");
        TORCH_CHECK(fake_data.device() == device, "Fake data must be on the same device");

        // Input validation
        TORCH_CHECK(d_real_logits.dim() == 1, "Real logits must be 1D (batch_size)");
        TORCH_CHECK(d_fake_logits.dim() == 1, "Fake logits must be 1D (batch_size)");
        TORCH_CHECK(real_data.sizes() == fake_data.sizes(), "Real and fake data must have the same shape");
        TORCH_CHECK(d_real_logits.size(0) == d_fake_logits.size(0) && d_real_logits.size(0) == real_data.size(0),
                    "Batch sizes must match");
        TORCH_CHECK(mode == "discriminator" || mode == "generator", "Mode must be 'discriminator' or 'generator'");
        TORCH_CHECK(lambda_gp >= 0, "Gradient penalty weight must be non-negative");

        // Generator loss: -mean(D(fake))
        if (mode == "generator")
        {
            return -d_fake_logits.mean();
        }

        // Discriminator loss: mean(D(fake)) - mean(D(real)) + lambda_gp * gradient_penalty
        torch::Tensor base_loss = d_fake_logits.mean() - d_real_logits.mean();

        // Compute gradient penalty
        int batch_size = real_data.size(0);
        // Generate interpolation coefficients
        torch::Tensor alpha = torch::rand({batch_size, 1, 1, 1}, torch::TensorOptions().device(device)).
            expand_as(real_data);
        // Interpolate between real and fake data, detaching to ensure interpolates is a leaf tensor
        torch::Tensor interpolates = alpha * real_data.detach() + (1.0 - alpha) * fake_data.detach();
        interpolates = interpolates.set_requires_grad(true);

        // Compute discriminator output for interpolated samples
        torch::Tensor d_interpolates = std::any_cast<torch::Tensor>(discriminator->forward({interpolates}));
        TORCH_CHECK(d_interpolates.dim() == 1, "Discriminator output for interpolates must be 1D (batch_size)");

        // Compute gradients of discriminator output w.r.t. interpolated inputs
        torch::Tensor gradients = torch::autograd::grad(
            {d_interpolates}, {interpolates}, {torch::ones_like(d_interpolates)},
            /*create_graph=*/true, /*retain_graph=*/true
        )[0];

        // Compute gradient norm
        gradients = gradients.view({batch_size, -1});
        torch::Tensor gradient_norm = gradients.norm(2, 1);
        torch::Tensor gradient_penalty = lambda_gp * torch::pow(gradient_norm - 1.0, 2).mean();

        // Combine base loss and gradient penalty
        return base_loss + gradient_penalty;
    }

    auto WGANGPLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        // return xt::losses::wgan_gp_loss(torch::zeros(10), torch::zeros(10), torch::zeros(10), torch::zeros(10),
        //                                 nullptr);
    }
}
