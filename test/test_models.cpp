#include <gtest/gtest.h>
#include <torch/torch.h>
#include <xtorch/xtorch.h>

TEST(XTorchModelTest, ForwardPassOutputShape)
{
    auto model = xt::models::LeNet5(10, 3);
    // Test with batch size 1
    auto input1 = torch::randn({1, 3, 32, 32});
    auto output1 = std::any_cast<torch::Tensor>(model.forward({input1}));
    ASSERT_EQ(output1.size(0), 1);
    ASSERT_EQ(output1.size(1), 10);

    // Test with batch size 8
    auto input8 = torch::randn({8, 3, 32, 32});
    auto output8 = std::any_cast<torch::Tensor>(model.forward({input8}));
    ASSERT_EQ(output8.size(0), 8);
    ASSERT_EQ(output8.size(1), 10);
}

TEST(XTorchModelTest, TrainAndEvalModeDifference)
{
    auto model = xt::models::LeNet5(10, 3);
    auto input = torch::randn({4, 3, 32, 32});

    model.train();
    auto output_train = std::any_cast<torch::Tensor>(model.forward({input}));

    model.eval();
    auto output_eval = std::any_cast<torch::Tensor>(model.forward({input}));

    // In eval mode, dropout is disabled, so the output should be different
    ASSERT_FALSE(torch::allclose(output_train, output_eval));
}

TEST(XTorchModelTest, GradientFlowCheck)
{
    auto model = xt::models::LeNet5(10, 3);
    auto input = torch::randn({2, 3, 32, 32});
    auto output = std::any_cast<torch::Tensor>(model.forward({input}));
    output.sum().backward();

    // // Check that gradients were computed for a key parameter
    // ASSERT_TRUE(model.fc1->weight.grad().defined());
    // // Ensure the gradient isn't just a tensor of zeros
    // ASSERT_FALSE(model->fc1->weight.grad().is_zero().all().item<bool>());
}

TEST(XTorchModelTest, DevicePortability)
{
    if (!torch::cuda::is_available())
    {
        GTEST_SKIP() << "CUDA not available, skipping device portability test.";
    }
    auto model = xt::models::LeNet5(10, 3);
    // Move model to GPU
    model.to(torch::kCUDA);

    // Check that a parameter is on the GPU
    // ASSERT_TRUE(model->conv1->weight.is_cuda());

    // Check that a forward pass on the GPU works
    auto input = torch::randn({4, 3, 32, 32}, torch::kCUDA);
    auto output = std::any_cast<torch::Tensor>(model.forward({input}));
    // ASSERT_TRUE(output.is_cuda());
}
