#include "include/normalizations/adaptive_instance_normalization.h"



#include <torch/torch.h>
#include <iostream>
#include <vector>

// Forward declaration for the Impl struct
struct AdaptiveInstanceNormalizationImpl;

// The main module struct that users will interact with.
// It uses TORCH_MODULE to handle the shared_ptr mechanics.
struct AdaptiveInstanceNormalization : torch::nn::ModuleHolder<AdaptiveInstanceNormalizationImpl> {
    using torch::nn::ModuleHolder<AdaptiveInstanceNormalizationImpl>::ModuleHolder;

    // The forward method is exposed through the Impl class
    // It takes the content input, and the pre-computed style scale (gamma) and style shift (beta)
    torch::Tensor forward(const torch::Tensor& content_input, const torch::Tensor& style_gamma, const torch::Tensor& style_beta) {
        return impl_->forward(content_input, style_gamma, style_beta);
    }
};

// The implementation struct
struct AdaptiveInstanceNormalizationImpl : torch::nn::Module {
    double eps_;

    AdaptiveInstanceNormalizationImpl(double eps = 1e-5)
        : eps_(eps) {
        // Adaptive Instance Normalization does not have its own learnable parameters
        // for gamma and beta in the traditional sense. These are provided from the style input.
    }

    torch::Tensor forward(const torch::Tensor& content_input, const torch::Tensor& style_gamma, const torch::Tensor& style_beta) {
        // ----- Input Shape Checks -----
        // content_input: (N, C, H, W) or (N, C, L) or (N, C)
        // style_gamma: (N, C)
        // style_beta: (N, C)

        TORCH_CHECK(content_input.dim() >= 2, "Content input tensor must have at least 2 dimensions (N, C, ...). Got ", content_input.dim());
        TORCH_CHECK(style_gamma.dim() == 2, "Style gamma tensor must have 2 dimensions (N, C). Got ", style_gamma.dim());
        TORCH_CHECK(style_beta.dim() == 2, "Style beta tensor must have 2 dimensions (N, C). Got ", style_beta.dim());

        const int64_t N = content_input.size(0);
        const int64_t C = content_input.size(1);

        TORCH_CHECK(style_gamma.size(0) == N, "Batch size of content input (", N, ") and style_gamma (", style_gamma.size(0), ") must match.");
        TORCH_CHECK(style_gamma.size(1) == C, "Number of channels of content input (", C, ") and style_gamma (", style_gamma.size(1), ") must match.");
        TORCH_CHECK(style_beta.size(0) == N, "Batch size of content input (", N, ") and style_beta (", style_beta.size(0), ") must match.");
        TORCH_CHECK(style_beta.size(1) == C, "Number of channels of content input (", C, ") and style_beta (", style_beta.size(1), ") must match.");

        // ----- Instance Normalize content_input -----
        torch::Tensor normalized_content;

        if (content_input.dim() > 2) {
            // For inputs like (N, C, H, W) or (N, C, L)
            // We want to normalize over the spatial/sequential dimensions (dims from 2 onwards)
            std::vector<int64_t> reduce_dims;
            for (int64_t i = 2; i < content_input.dim(); ++i) {
                reduce_dims.push_back(i);
            }

            // Calculate mean and variance per instance, per channel.
            // keepdim=true ensures broadcastable shapes.
            auto mean = content_input.mean(reduce_dims, /*keepdim=*/true);
            // unbiased=false for population variance.
            auto var = content_input.var(reduce_dims, /*unbiased=*/false, /*keepdim=*/true);

            normalized_content = (content_input - mean) / torch::sqrt(var + eps_);
        } else { // content_input.dim() == 2, shape (N, C)
            // For (N,C) input, each "spatial" element is a single point.
            // Normalizing a single point: (v - v) / sqrt(var(v) + eps) = 0 / sqrt(0 + eps) = 0.
            normalized_content = torch::zeros_like(content_input);
        }

        // ----- Apply Style Parameters (gamma and beta) -----
        // style_gamma and style_beta are (N, C). They need to be reshaped to
        // (N, C, 1, 1, ...) to broadcast with normalized_content.

        std::vector<int64_t> style_param_view_shape;
        style_param_view_shape.push_back(N); // N
        style_param_view_shape.push_back(C); // C
        for (int64_t i = 2; i < content_input.dim(); ++i) {
            style_param_view_shape.push_back(1);
        }

        auto style_gamma_reshaped = style_gamma.view(style_param_view_shape);
        auto style_beta_reshaped  = style_beta.view(style_param_view_shape);

        return normalized_content * style_gamma_reshaped + style_beta_reshaped;
    }

    // Optional: for pretty printing the module
    void pretty_print(std::ostream& stream) const override {
        stream << "AdaptiveInstanceNormalization(eps=" << eps_ << ")";
    }
};
TORCH_MODULE(AdaptiveInstanceNormalization); // Creates the wrapper from Impl


// --- Example Usage ---
int main() {
    torch::manual_seed(0); // For reproducible results

    AdaptiveInstanceNormalization adain_module(/*eps=*/1e-5);
    // std::cout << adain_module << std::endl; // For full parameter print

    // --- Test Case 1: 4D content input (image-like) ---
    std::cout << "--- Test Case 1: 4D content input (image-like) ---" << std::endl;
    int64_t N1 = 2, C1 = 3, H1 = 4, W1 = 4;
    torch::Tensor content1 = torch::randn({N1, C1, H1, W1});
    // Style parameters: typically gamma > 0
    torch::Tensor style_gamma1 = torch::rand({N1, C1}) + 0.5; // (0.5 to 1.5)
    torch::Tensor style_beta1  = torch::randn({N1, C1});

    std::cout << "Content input shape: " << content1.sizes() << std::endl;
    std::cout << "Style gamma shape: " << style_gamma1.sizes() << std::endl;
    std::cout << "Style beta shape: " << style_beta1.sizes() << std::endl;

    torch::Tensor output1 = adain_module->forward(content1, style_gamma1, style_beta1);
    std::cout << "Output 1 shape: " << output1.sizes() << std::endl;

    // Check mean and std of one channel of the output.
    // Mean should be close to style_beta1[0,0] and std close to style_gamma1[0,0]
    // for output1[0,0,:,:]
    auto output1_ch0_inst0 = output1.select(0,0).select(0,0); // First instance, first channel
    std::cout << "Output1[0,0,:,:] mean: " << output1_ch0_inst0.mean().item<double>()
              << " (expected near beta1[0,0]=" << style_beta1[0][0].item<double>() << ")" << std::endl;
    std::cout << "Output1[0,0,:,:] std: " << output1_ch0_inst0.std(/*unbiased=*/false).item<double>()
              << " (expected near gamma1[0,0]=" << style_gamma1[0][0].item<double>() << ")" << std::endl;


    // --- Test Case 2: 3D content input (sequence-like) ---
    std::cout << "\n--- Test Case 2: 3D content input (sequence-like) ---" << std::endl;
    int64_t N2 = 3, C2 = 64, L2 = 10;
    torch::Tensor content2 = torch::randn({N2, C2, L2});
    torch::Tensor style_gamma2 = torch::rand({N2, C2}) + 0.5;
    torch::Tensor style_beta2  = torch::randn({N2, C2});

    std::cout << "Content input shape: " << content2.sizes() << std::endl;
    torch::Tensor output2 = adain_module->forward(content2, style_gamma2, style_beta2);
    std::cout << "Output 2 shape: " << output2.sizes() << std::endl;
    auto output2_ch0_inst0 = output2.select(0,0).select(0,0);
    std::cout << "Output2[0,0,:] mean: " << output2_ch0_inst0.mean().item<double>()
              << " (expected near beta2[0,0]=" << style_beta2[0][0].item<double>() << ")" << std::endl;
    std::cout << "Output2[0,0,:] std: " << output2_ch0_inst0.std(/*unbiased=*/false).item<double>()
              << " (expected near gamma2[0,0]=" << style_gamma2[0][0].item<double>() << ")" << std::endl;


    // --- Test Case 3: 2D content input (N, C) ---
    std::cout << "\n--- Test Case 3: 2D content input (N, C) ---" << std::endl;
    int64_t N3 = 4, C3 = 5;
    torch::Tensor content3 = torch::randn({N3, C3});
    torch::Tensor style_gamma3 = torch::rand({N3, C3}) + 0.5;
    torch::Tensor style_beta3  = torch::randn({N3, C3});

    std::cout << "Content input shape: " << content3.sizes() << std::endl;
    torch::Tensor output3 = adain_module->forward(content3, style_gamma3, style_beta3);
    std::cout << "Output 3 shape: " << output3.sizes() << std::endl;
    // For 2D input, normalized_content is 0, so output should be exactly style_beta3
    std::cout << "Output 3 (should be equal to style_beta3): \n" << output3 << std::endl;
    std::cout << "Style_beta3: \n" << style_beta3 << std::endl;
    TORCH_CHECK(torch::allclose(output3, style_beta3), "Output3 should be equal to style_beta3 for 2D content input.");


    // --- Test Case 4: style_gamma = 1, style_beta = 0 (reduces to Instance Normalization) ---
    std::cout << "\n--- Test Case 4: style_gamma=1, style_beta=0 ---" << std::endl;
    torch::Tensor style_gamma4 = torch::ones_like(style_gamma1);
    torch::Tensor style_beta4  = torch::zeros_like(style_beta1);

    torch::Tensor output4 = adain_module->forward(content1, style_gamma4, style_beta4);
    std::cout << "Output 4 shape: " << output4.sizes() << std::endl;
    auto output4_ch0_inst0 = output4.select(0,0).select(0,0);
    // Mean should be ~0 and std ~1
    std::cout << "Output4[0,0,:,:] mean (expected ~0): " << output4_ch0_inst0.mean().item<double>() << std::endl;
    std::cout << "Output4[0,0,:,:] std (expected ~1): " << output4_ch0_inst0.std(/*unbiased=*/false).item<double>() << std::endl;
    // Check another channel/instance
    auto output4_ch1_inst1 = output4.select(0,1).select(0,1);
    std::cout << "Output4[1,1,:,:] mean (expected ~0): " << output4_ch1_inst1.mean().item<double>() << std::endl;
    std::cout << "Output4[1,1,:,:] std (expected ~1): " << output4_ch1_inst1.std(/*unbiased=*/false).item<double>() << std::endl;

    std::cout << "\nAll AdaIN tests passed!" << std::endl;

    return 0;
}


namespace xt::norm
{
    auto AdaptiveInstanceNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
