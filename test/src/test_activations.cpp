#include <gtest/gtest.h>
#include <torch/torch.h>
#include <xtorch/xtorch.h>

TEST(XTorchActivationTest, GeluGoldenValues) {
    // Test a vector of known inputs
    auto input = torch::tensor({-2.0, -1.0, 0.0, 1.0, 2.0});
    // auto output = xt::activations::gelu(input);

    // Expected values calculated from a trusted implementation (e.g., Hugging Face Transformers)
    auto expected = torch::tensor({-0.0455, -0.1588, 0.0, 0.8412, 1.9545});

    // ASSERT_TRUE(torch::allclose(output, expected, 1e-4, 1e-4));
}

TEST(XTorchActivationTest, GeluGradientCheck) {
    // Ensure that autograd works correctly through the function
    auto input = torch::tensor({-0.5, 0.5}, torch::requires_grad());
    // auto output = xt::activations::gelu(input);
    // output.sum().backward();

    ASSERT_TRUE(input.grad().defined());
    ASSERT_EQ(input.grad().sizes(), input.sizes());

    // Manual derivative of GELU at 0.5 is approx 0.849
    // Manual derivative of GELU at -0.5 is approx 0.150
    auto expected_grad = torch::tensor({0.1503, 0.8496});
    ASSERT_TRUE(torch::allclose(input.grad(), expected_grad, 1e-4, 1e-4));
}