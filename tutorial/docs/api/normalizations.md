# Normalization Layers

Normalization layers are a crucial component in modern deep neural networks. They help stabilize the learning process, reduce internal covariate shift, and often lead to faster convergence and better generalization.

## Standard LibTorch Normalizations

LibTorch provides a solid foundation with the most widely-used normalization layers. These are the go-to choices for many standard architectures.

**Common Examples:**
- `torch::nn::BatchNorm1d`, `torch::nn::BatchNorm2d`, `torch::nn::BatchNorm3d`
- `torch::nn::LayerNorm`
- `torch::nn::InstanceNorm1d`, `torch::nn::InstanceNorm2d`, `torch::nn::InstanceNorm3d`
- `torch::nn::GroupNorm`
- `torch::nn::LocalResponseNorm`

For detailed usage of these standard layers, please refer to the [official PyTorch C++ API documentation](https://pytorch.org/cppdocs/api/namespace_torch__nn.html).

---

## xTorch Extended Normalizations

To enable research and development with cutting-edge architectures, xTorch provides an extensive collection of advanced and specialized normalization techniques proposed in the literature.

These implementations allow you to move beyond `BatchNorm` and experiment with alternatives like `EvoNorms`, `FilterResponseNormalization`, or `WeightStandardization` with ease.

### Usage

All xTorch normalization layers are located in the `xt::normalizations` namespace. They are implemented as `torch::nn::Module`s and can be integrated directly into your model definitions, just like any standard `torch::nn` module.

```cpp
#include <xtorch/xtorch.hh>

// A simple convolutional block using Filter Response Normalization
struct ConvBlock : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};
    // Instantiate the FRN layer for 64 channels
    xt::normalizations::FilterResponseNormalization frn;
    xt::activations::Mish mish;

    ConvBlock(int in_channels, int out_channels)
        : frn(xt::normalizations::FilterResponseNormalizationOptions(out_channels))
    {
        conv = register_module("conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)
        ));
        // Register the normalization layer
        register_module("frn", frn);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv(x);
        // Apply normalization, then activation
        x = frn(x);
        x = mish(x);
        return x;
    }
};
```

!!! warning "Training vs. Evaluation Mode"
Like `BatchNorm`, many normalization layers have different behaviors during training (e.g., updating running statistics) and evaluation. Always remember to call `model.train()` or `model.eval()` to switch the model to the appropriate mode.

### Available Normalization Layers

Below is the list of normalization techniques available in the `xt::normalizations` module.

| | | | |
|---|---|---|---|
| `ActivationNormalization` | `AdaptiveInstanceNormalization` | `AttentiveNormalization` | `BatchChannelNormalization` |
| `CincFlow` | `ConditionalBatchNormalization` | `ConditionalInstanceNormalization` | `CosineNormalization` |
| `CrossNorm` | `DecorrelatedBatchNormalization` | `EvoNorms` | `FilterResponseNormalization` |
| `GradientNormalization` | `InPlaceABN` | `InstanceLevelMetaNormalization` | `LayerScale` |
| `LocalContrastNormalization` | `MixtureNormalization` | `ModeNormalization` | `MPNNormalization` |
| `OnlineNormalization` | `PixelNormalization` | `PowerNormalization` | `ReZero` |
| `SABN` | `SelfNorm` | `SPADE` | `SparseSwitchableNormalization` |
| `SpectralNormalization` | `SRN` | `SwitchableNormalization` | `SyncBN` |
| `VirtualBatchNormalization` | `WeightDemodulation` | `WeightNormalization` | `WeightStandardization` |


!!! info "Constructor Options"
Many of these layers require specific arguments during construction, such as the number of channels or features. These are configured via an `Options` struct passed to the constructor. Please refer to the specific header file in `<xtorch/normalizations/>` for details on the available settings for each layer.
