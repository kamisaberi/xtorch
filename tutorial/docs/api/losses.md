# Loss Functions

A loss function (or criterion) is a fundamental component of training a neural network. It calculates a single value that measures how well the model's prediction matches the target label. The goal of training is to minimize this value.

## Standard LibTorch Losses

LibTorch provides a comprehensive set of standard loss functions that are sufficient for most common tasks. These can be used directly within any xTorch project.

**Common Examples:**
- `torch::nn::MSELoss`: Mean Squared Error, for regression.
- `torch::nn::L1Loss`: Mean Absolute Error, for regression.
- `torch::nn::CrossEntropyLoss`: For multi-class classification.
- `torch::nn::BCELoss`: Binary Cross Entropy, for binary classification.
- `torch::nll_loss`: Negative Log Likelihood Loss, often used with `LogSoftmax`.

For a complete list and usage instructions, please refer to the [official PyTorch C++ API documentation](https://pytorch.org/cppdocs/api/namespace_torch__nn.html).

---

## xTorch Extended Losses

For more advanced or specialized tasks—such as object detection, metric learning, or dealing with class imbalance—standard loss functions may not be optimal. To address this, xTorch provides a rich library of modern and specialized loss functions from the research literature.

These implementations are ready to use, allowing you to easily experiment with advanced training objectives.

### Usage

All xTorch loss functions are implemented as `torch::nn::Module`s and are located in the `xt::losses` namespace. They can be instantiated and used just like any standard `torch::nn` module.

They integrate seamlessly with the `xt::Trainer`.

```cpp
#include <xtorch/xtorch.hh>

int main() {
    // Assume we have model outputs and targets
    auto output = torch::randn({16, 10}); // Batch of 16, 10 classes
    auto target = torch::randint(0, 10, {16}); // 16 target labels

    // --- Standalone Usage ---
    // Instantiate a loss function with its options
    xt::losses::FocalLoss focal_loss(xt::losses::FocalLossOptions().gamma(2.0));
    // Calculate the loss
    torch::Tensor loss = focal_loss(output, target);
    std::cout << "Calculated Focal Loss: " << loss.item<float>() << std::endl;


    // --- Integration with xt::Trainer ---
    xt::Trainer trainer;
    // The trainer can accept any callable, including our instantiated module
    trainer.set_loss_fn(focal_loss);

    // Now, when trainer.fit() is called, it will use the FocalLoss internally.
    // trainer.fit(model, data_loader, nullptr, device);
}
```

### Available Loss Functions

Below is the comprehensive list of specialized loss functions available in the `xt::losses` module.

| | | | |
|---|---|---|---|
| `AdaptiveLoss` | `ArcFaceLoss` | `BalancedL1Loss` | `CycleConsistencyLoss`|
| `DHELLoss` | `DiceBCELoss` | `DiceLoss` | `DSAMLoss` |
| `DualSoftmaxLoss` | `DynamicSmoothL1Loss` | `EarlyExitingLoss` | `ElasticFace` |
| `FlipLoss` | `FocalLoss` | `GANHingeLoss` | `GANLeastSquaresLoss`|
| `GeneralizedFocalLoss`|`GHMCLoss`| `GHMRLoss` | `HappierLoss` |
| `HBMLoss` | `InfoNCELoss` | `LovaszSoftmaxLoss` | `MetricMixupLoss` |
| `MultiLoss` | `NTXentLoss` | `ObjectAwareLoss` | `PIOULoss` |
| `ProxyAnchorLoss` | `RankBasedLoss` | `SeesawLoss` | `SelfAdjustingSmoothL1Loss` |
| `SupervisedContrastiveLoss` | `TripletEntropyLoss` | `TripletLoss`|`UnsupervisedFeatureLoss`|
| `UPITLoss` | `VarifocalLoss` | `WGANGPLoss` | `ZLPRLoss` |


!!! info "Constructor Options"
Many of these loss functions have tunable hyperparameters (like `gamma` in `FocalLoss` or `margin` in `TripletLoss`). These are configured via an `Options` struct passed to the constructor. Please refer to the specific header file in `<xtorch/losses/>` for details on the available settings.
