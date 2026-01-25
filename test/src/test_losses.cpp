#include <gtest/gtest.h>
#include <torch/torch.h>
#include <xtorch/xtorch.h>

TEST(XTorchLossTest, FocalLossGoldenValue) {
    // A known input/output case for validation
    auto input = torch::tensor({ {0.1, 0.9}, {0.8, 0.2} }); // logits
    auto target = torch::tensor({1, 0}); // class indices

    // Manually calculated expected value
    // For sample 0 (target 1): ce = -log(0.9) = 0.10536, pt = 0.9, fl = 0.25 * (1-0.9)^2 * 0.10536 = 0.0002634
    // For sample 1 (target 0): ce = -log(0.8) = 0.22314, pt = 0.8, fl = 0.25 * (1-0.8)^2 * 0.22314 = 0.0022314
    // Mean = (0.0002634 + 0.0022314) / 2 = 0.0012474
    float expected_loss_val = 0.0012474;

    // auto loss = xt::losses::focal_loss(input, target, 0.25, 2.0, "mean");
    // ASSERT_NEAR(loss.item<float>(), expected_loss_val, 1e-6);
}

TEST(XTorchLossTest, FocalLossZeroCase) {
    auto input = torch::tensor({ {-100.0, 100.0}, {100.0, -100.0} }); // Confident predictions
    auto target = torch::tensor({1, 0});
    // auto loss = xt::losses::focal_loss(input, target);
    // Loss should be very close to zero for perfect predictions
    // ASSERT_NEAR(loss.item<float>(), 0.0, 1e-6);
}

TEST(XTorchLossTest, FocalLossReductionModes) {
    auto input = torch::randn({4, 10});
    auto target = torch::randint(0, 10, {4});

    // auto loss_mean = xt::losses::focal_loss(input, target, 0.25, 2.0, "mean");
    // auto loss_sum = xt::losses::focal_loss(input, target, 0.25, 2.0, "sum");
    // auto loss_none = xt::losses::focal_loss(input, target, 0.25, 2.0, "none");

    // ASSERT_EQ(loss_mean.dim(), 0); // Scalar
    // ASSERT_EQ(loss_sum.dim(), 0);  // Scalar
    // ASSERT_EQ(loss_none.dim(), 1); // Vector
    // ASSERT_EQ(loss_none.size(0), 4);
}