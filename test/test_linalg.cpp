#include <gtest/gtest.h>
#include <torch/torch.h>
// #include <xtorch/xtorch.h>
#include  "my_lib.h"

// Test case for the double_tensor function
TEST(MyLibTest, DoubleTensor) {
    // Create an input tensor
    torch::Tensor input = torch::tensor({1.0, 2.0, 3.0});

    // Call the function to be tested
    torch::Tensor output = double_tensor(input);

    // Create the expected output tensor
    torch::Tensor expected_output = torch::tensor({2.0, 4.0, 6.0});

    // Check if the output is close to the expected output
    ASSERT_TRUE(torch::allclose(output, expected_output));
}

// Another test case to demonstrate a failure
TEST(MyLibTest, DoubleTensorFailure) {
    torch::Tensor input = torch::tensor({1.0, 2.0, 3.0});
    torch::Tensor output = double_tensor(input);
    torch::Tensor unexpected_output = torch::tensor({2.0, 5.0, 6.0});

    // This assertion is expected to fail
    EXPECT_TRUE(torch::allclose(output, unexpected_output));
}