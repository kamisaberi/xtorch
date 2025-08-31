# Dropouts

Dropout is a powerful regularization technique used to prevent overfitting in neural networks. During training, it randomly sets a fraction of input units to zero at each update, which helps the model learn more robust features that are not overly dependent on any single neuron.

## Standard LibTorch Dropouts

LibTorch provides the most common dropout implementations, which are fully supported and can be used in any xTorch model.

**Standard Modules:**
- `torch::nn::Dropout`: Randomly zeroes entire elements.
- `torch::nn::Dropout2d`: Randomly zeroes entire channels of a 2D feature map.
- `torch::nn::Dropout3d`: Randomly zeroes entire channels of a 3D feature map.

For detailed usage, please see the [official PyTorch C++ API documentation](https://pytorch.org/cppdocs/api/namespace_torch__nn.html).

---

## xTorch Extended Dropouts

Experimenting with different regularization strategies is key to achieving state-of-the-art performance. To facilitate this, xTorch provides a comprehensive library of advanced and specialized dropout variants proposed in recent research.

These implementations allow you to easily swap out standard dropout with more sophisticated techniques like `DropBlock`, `ScheduledDropPath`, or `VariationalDropout`.

### Usage

All xTorch dropout modules are located in the `xt::dropouts` namespace. They are implemented as `torch::nn::Module`s and can be integrated into your models just like `torch::nn::Dropout`.

Most dropout layers require a dropout probability `p` during construction.

```cpp
#include <xtorch/xtorch.h>

// Define a model with an advanced dropout variant
struct MyModel : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    // Instantiate DropBlock with a probability of 0.5 and block size of 7
    xt::dropouts::DropBlock drop_block;

    MyModel()
        : drop_block(xt::dropouts::DropBlockOptions(0.5).block_size(7))
    {
        fc1 = register_module("fc1", torch::nn::Linear(784, 128));
        // Register the dropout module
        register_module("drop_block", drop_block);
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        // Apply dropout during training
        x = drop_block(x);
        x = fc2(x);
        return x;
    }
};

int main() {
    MyModel model;
    // Set the model to training mode to enable dropout
    model.train();
    auto input = torch::randn({32, 784});
    auto output = model.forward(input);
    std::cout << "Output shape: " << output.sizes() << std::endl;
}
```

!!! warning "Training vs. Evaluation Mode"
Dropout is only active during training. Always remember to call `model.eval()` before running inference to automatically disable all dropout layers in the model.

### Available Dropouts

Below is the list of specialized dropout implementations available in the `xt::dropouts` module.

| | | | |
|---|---|---|---|
| `AdaptiveDropout` | `AttentionDropout` | `AutoDropout` | `BandDropout` |
| `CheckerboardDropout` | `ConcreteDropout` | `CurriculumDropout` | `DropBlock` |
| `DropConnect` | `DropPath` | `DropPathway` | `EarlyDropout` |
| `EmbeddingDropout` | `FraternalDropout` | `GaussianDropout` | `GradDrop` |
| `LayerDrop` | `MonteCarloDropout` | `RecurrentDropout` | `RNNDrop` |
| `ScheduledDropPath` | `SensorDropout` | `ShakeDrop` | `SpatialDropout` |
| `SpectralDropout` | `TargetedDropout` | `TemporalDropout` | `VariationalDropout` |
| `VariationalGaussianDropout` | `ZoneOut` |

!!! info "Constructor Options"
Many dropout variants have unique parameters (e.g., `block_size` for `DropBlock`). Refer to the corresponding header file in `<xtorch/dropouts/>` for the specific `Options` struct and available settings.
