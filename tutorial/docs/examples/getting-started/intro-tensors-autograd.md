# Getting Started: Introduction to Tensors and Autograd

Before building complex neural networks, it's essential to understand the two fundamental building blocks of xTorch and PyTorch/LibTorch: **Tensors** and **Autograd**.

1.  **`torch::Tensor`**: The primary data structure. It's a multi-dimensional array, similar to NumPy's `ndarray`, but with the crucial ability to run on GPUs and track computational history.
2.  **`torch::autograd`**: The automatic differentiation engine. It records all operations performed on tensors to automatically calculate gradients, which is the cornerstone of how neural networks learn.

This tutorial will walk you through creating tensors, performing operations, and using `autograd` to compute gradients.

---

## 1. Tensors

A tensor is a generalization of vectors and matrices to an arbitrary number of dimensions. In xTorch, all inputs, outputs, and model parameters are represented as `torch::Tensor` objects.

### Creating Tensors

You can create tensors in several ways.

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    // Create an uninitialized 5x3 tensor
    torch::Tensor x = torch::empty({5, 3});
    std::cout << "Empty Tensor:\n" << x << std::endl;

    // Create a randomly initialized 5x3 tensor
    torch::Tensor r = torch::rand({5, 3});
    std::cout << "Random Tensor:\n" << r << std::endl;

    // Create a 5x3 tensor filled with zeros, of type long
    torch::Tensor z = torch::zeros({5, 3}, torch::kLong);
    std::cout << "Zeros Tensor:\n" << z << std::endl;

    // Create a tensor directly from data
    torch::Tensor d = torch::tensor({{1, 2}, {3, 4}});
    std::cout << "Tensor from data:\n" << d << std::endl;

    // Get tensor attributes
    std::cout << "Size of d: " << d.sizes() << std::endl;
    std::cout << "Dtype of z: " << z.dtype() << std::endl;
}
```

### Tensor Operations

xTorch supports a comprehensive library of mathematical operations.

```cpp
// Create two tensors
torch::Tensor a = torch::ones({2, 2});
torch::Tensor b = torch::randn({2, 2});

// Addition
torch::Tensor c = a + b;
// or torch::add(a, b);
std::cout << "a + b:\n" << c << std::endl;

// In-place addition (modifies the tensor)
a.add_(b);
std::cout << "a after in-place add:\n" << a << std::endl;

// Standard array indexing is supported
std::cout << "First column of a:\n" << a.index({torch::indexing::Slice(), 0}) << std::endl;
```

---

## 2. Autograd: Automatic Differentiation

This is the magic behind how neural networks learn. `autograd` tracks all operations on tensors to build a dynamic computation graph. When you have a final scalar value (like a loss), you can call `.backward()` on it, and `autograd` will automatically compute the gradients of that value with respect to any "leaf" tensors.

### Tracking Gradients

To track computations, a tensor must have its `requires_grad` property set to `true`.

```cpp
// Create two tensors that we want to compute gradients for
auto w = torch::randn({3, 1}, torch::requires_grad());
auto b = torch::randn({1}, torch::requires_grad());

std::cout << "w:\n" << w << std::endl;
std::cout << "b:\n" << b << std::endl;

// Create another tensor for input data (we don't need gradients for this)
auto x = torch::randn({1, 3});

// Perform a forward pass (a simple linear operation)
// autograd builds the computation graph: y depends on w and b
auto y = torch::matmul(x, w) + b;

// Now, let's assume 'y' is the output of a model, and we have a target value.
// We calculate a scalar loss.
auto target = torch::ones({1, 1});
auto loss = torch::mse_loss(y, target);

std::cout << "Final loss: " << loss.item<float>() << std::endl;
```

### Computing Gradients with `.backward()`

The `loss` tensor now knows its entire history. When we call `loss.backward()`, `autograd` traverses this history backward and computes the gradient of the loss with respect to `w` and `b`.

```cpp
// Before backward pass, gradients are null
std::cout << "w.grad() before backward: " << w.grad() << std::endl;

// Compute the gradients
loss.backward();

// After backward pass, the .grad() attribute is populated
std::cout << "w.grad() after backward:\n" << w.grad() << std::endl;
std::cout << "b.grad() after backward:\n" << b.grad() << std::endl;
```
The optimizer would then use these `.grad()` values to update the weights `w` and `b`.

### Disabling Gradient Tracking

During inference, or when you are certain you won't need to compute gradients, it's more efficient to disable tracking. You can do this with a `torch::NoGradGuard`.

```cpp
{
    torch::NoGradGuard no_grad;
    // Any operation inside this block will not be tracked
    auto y_pred = torch::matmul(x, w) + b;
    std::cout << "y_pred requires_grad: " << y_pred.requires_grad() << std::endl; // Will be false
}
```

---

## Full C++ Code

Below is the complete, runnable source code for this example. The original file is located at `getting_started/introduction_to_xtorch__tensors_and_autograd.cpp`.

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "--- Tensors ---" << std::endl;

    // Create tensors
    torch::Tensor r = torch::rand({2, 2});
    std::cout << "Random Tensor:\n" << r << std::endl;
    torch::Tensor z = torch::zeros({2, 2});
    std::cout << "Zeros Tensor:\n" << z << std::endl;

    // Tensor operations
    torch::Tensor c = r + z;
    std::cout << "r + z:\n" << c << std::endl;

    std::cout << "\n--- Autograd ---" << std::endl;

    // Create tensors for which we want gradients
    auto w = torch::randn({3, 1}, torch::requires_grad());
    auto b = torch::randn({1}, torch::requires_grad());
    std::cout << "Initial w:\n" << w << std::endl;
    std::cout << "Initial b:\n" << b << std::endl;

    // Input data
    auto x = torch::tensor({{1.0, 2.0, 3.0}});

    // Forward pass
    auto y = torch::matmul(x, w) + b;

    // Calculate a scalar loss
    auto target = torch::tensor({{10.0}});
    auto loss = torch::mse_loss(y, target);
    std::cout << "Calculated Loss: " << loss.item<float>() << std::endl;

    // Check gradients before backward pass (they will be null)
    std::cout << "w.grad() before backward: " << w.grad() << std::endl;

    // Compute gradients
    loss.backward();

    // Check gradients after backward pass
    std::cout << "w.grad() after backward:\n" << w.grad() << std::endl;
    std::cout << "b.grad() after backward:\n" << b.grad() << std::endl;

    // Demonstrate NoGradGuard
    std::cout << "\n--- NoGradGuard ---" << std::endl;
    std::cout << "y requires_grad: " << y.requires_grad() << std::endl;
    {
        torch::NoGradGuard no_grad;
        auto y_pred = torch::matmul(x, w) + b;
        std::cout << "y_pred requires_grad: " << y_pred.requires_grad() << std::endl;
    }

    return 0;
}
```

Now that you understand the basics of Tensors and Autograd, you are ready to see how they are used to [build and train your first neural network](building-simple-nn.md).