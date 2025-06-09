#include "include/losses/gan_least_squares_loss.h"

namespace xt::losses
{
    /**
     * Least Squares GAN Loss function for GAN training.
     * @param d_real_logits Discriminator logits for real samples, shape: [batch_size]
     * @param d_fake_logits Discriminator logits for fake samples, shape: [batch_size]
     * @param mode Specifies the loss type: "discriminator" or "generator"
     * @return Scalar tensor containing the Least Squares GAN Loss
     */
    torch::Tensor gan_least_squares_loss(const torch::Tensor& d_real_logits, const torch::Tensor& d_fake_logits,
                                         const std::string& mode)
    {
        // Ensure inputs are on the same device
        auto device = d_real_logits.device();
        TORCH_CHECK(d_fake_logits.device() == device, "Real and fake logits must be on the same device");

        // Input validation
        TORCH_CHECK(d_real_logits.dim() == 1, "Real logits must be 1D (batch_size)");
        TORCH_CHECK(d_fake_logits.dim() == 1, "Fake logits must be 1D (batch_size)");
        TORCH_CHECK(d_real_logits.size(0) == d_fake_logits.size(0), "Batch sizes must match");
        TORCH_CHECK(mode == "discriminator" || mode == "generator", "Mode must be 'discriminator' or 'generator'");

        if (mode == "discriminator")
        {
            // Discriminator loss: (D(real) - 1)^2 + (D(fake) - 0)^2
            torch::Tensor real_loss = torch::pow(d_real_logits - 1.0, 2);
            torch::Tensor fake_loss = torch::pow(d_fake_logits, 2);
            return (real_loss + fake_loss).mean();
        }
        else
        {
            // Generator loss: (D(fake) - 1)^2
            return torch::pow(d_fake_logits - 1.0, 2).mean();
        }
    }

    auto GANLeastSquaresLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::gan_least_squares_loss(torch::zeros(10));
    }
}
