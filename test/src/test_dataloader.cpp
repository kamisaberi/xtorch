#include <gtest/gtest.h>
#include <xtorch/xtorch.h>

TEST(XTorchDataloaderTest, CorrectSize) {
    auto train_dataset = xt::datasets::MNIST("./data" , xt::datasets::DataMode::TRAIN);
    auto test_dataset = xt::datasets::MNIST("./data" , xt::datasets::DataMode::TEST);
    ASSERT_EQ(train_dataset.size().value(), 100);
    ASSERT_EQ(test_dataset.size().value(), 20);
}

TEST(XTorchDataloaderTest, SampleShapeAndType) {
    auto dataset = xt::datasets::MNIST("./data" , xt::datasets::DataMode::TRAIN);
    auto sample = dataset.get(0);
    // Check data
    ASSERT_EQ(sample.data.dim(), 3);
    ASSERT_EQ(sample.data.size(0), 3); // Channels
    ASSERT_EQ(sample.data.size(1), 32); // Height
    ASSERT_EQ(sample.data.size(2), 32); // Width
    ASSERT_EQ(sample.data.dtype(), torch::kFloat32);

    // Check target
    ASSERT_EQ(sample.target.dim(), 0); // Scalar
    ASSERT_EQ(sample.target.dtype(), torch::kInt64);
}

TEST(XTorchDataloaderTest, OutOfBoundsAccess) {
    auto dataset = xt::datasets::MNIST("./data" , xt::datasets::DataMode::TRAIN);
    ASSERT_THROW(dataset.get(dataset.size().value()), std::exception);
}