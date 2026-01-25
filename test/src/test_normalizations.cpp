#include <gtest/gtest.h>
#include <torch/torch.h>
#include <xtorch/xtorch.h>

TEST(XTorchNormalizationTest, LarsSingleStep) {
    auto param = torch::tensor({1.0, 2.0, 3.0}, torch::requires_grad());
    auto grad = torch::tensor({0.1, 0.2, 0.3});
    // param.grad() = grad;
    //
    // auto optimizer = xt::optim::Lars({param}, 0.1); // lr = 0.1
    //
    // optimizer.step();
    //
    // // Manual calculation for a single SGD step: param_new = param_old - lr * grad
    // // {1.0 - 0.1 * 0.1, 2.0 - 0.1 * 0.2, 3.0 - 0.1 * 0.3}
    // // {1.0 - 0.01, 2.0 - 0.02, 3.0 - 0.03}
    // // {0.99, 1.98, 2.97}
    // auto expected = torch::tensor({0.99, 1.98, 2.97});
    //
    // ASSERT_TRUE(torch::allclose(param, expected));
}

TEST(XTorchNormalizationTest, ZeroGrad) {
    auto param = torch::tensor({1.0, 2.0, 3.0}, torch::requires_grad());
    // param.grad() = torch::ones_like(param);
    //
    // auto optimizer = xtorch::optim::Lars({param});
    // ASSERT_TRUE(param.grad().defined());
    //
    // optimizer.zero_grad();
    // ASSERT_FALSE(param.grad().defined());
}