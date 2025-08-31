# Regularization Techniques

Regularization refers to a collection of techniques designed to prevent a model from overfitting the training data. By adding a penalty for model complexity, regularization helps the model generalize better to unseen data.

## Standard Regularization in LibTorch

The most common forms of regularization are readily available when using LibTorch and are standard practice in deep learning.

1.  **Weight Decay (L2 Regularization)**: This is the most common technique. It is not a separate module but rather an option built directly into optimizers. You can enable it by setting the `weight_decay` parameter in the optimizer's options.

    ```cpp
    // Enable weight decay in the Adam optimizer
    torch::optim::Adam optimizer(
        model.parameters(),
        torch::optim::AdamOptions(1e-3).weight_decay(1e-4) // L2 penalty
    );
    ```

2.  **Dropout**: This technique randomly zeroes out activations during training. It is a powerful regularizer implemented as a set of modules. See the dedicated **[Dropouts](dropouts.md)** page for a comprehensive list of variants.

---

## xTorch Extended Regularization Techniques

Beyond weight decay and dropout, there is a wide range of explicit regularization methods that can be applied to activations, weights, or the loss function itself. xTorch provides a rich collection of these techniques, allowing for advanced experimentation.

### Usage

Most xTorch regularization techniques are implemented as `torch::nn::Module`s and are located in the `xt::regulariztions` namespace. They can be applied in your model's forward pass or used to wrap your loss function.

A common example is `LabelSmoothing`, which helps prevent the model from becoming overconfident in its predictions.

```cpp
#include <xtorch/xtorch.hh>

int main() {
    // Assume we have model outputs and targets
    auto logits = torch::randn({16, 10}); // Raw outputs (logits)
    auto targets = torch::randint(0, 10, {16});

    // 1. Define a standard loss function
    torch::nn::CrossEntropyLoss cross_entropy_loss;

    // 2. Instantiate the LabelSmoothing regularizer
    // Epsilon is the smoothing factor.
    xt::regulariztions::LabelSmoothing label_smoother(
        xt::regulariztions::LabelSmoothingOptions(0.1) // 10% smoothing
    );

    // 3. Apply label smoothing to the loss calculation
    // This typically involves combining the smoother's output with the standard loss.
    // The exact usage may vary, so always check the header.
    // For LabelSmoothing, it acts as a loss itself.
    torch::Tensor smoothed_loss = label_smoother(logits, targets);

    std::cout << "Standard Cross Entropy Loss: "
              << cross_entropy_loss(logits, targets).item<float>() << std::endl;
    std::cout << "Loss with Label Smoothing: "
              << smoothed_loss.item<float>() << std::endl;

    // 4. In a Trainer, you would set this as your loss function
    xt::Trainer trainer;
    trainer.set_loss_fn(label_smoother);
    // trainer.fit(...);
}
```

### Available Regularization Techniques

Below is the list of regularization modules available in the `xt::regulariztions` module.

| | | | |
|---|---|---|---|
| `ActivationRegularization` | `ALS` | `AuxiliaryBatchNormalization` | `BatchNuclearNormMaximization` |
| `DiscriminativeRegularization` | `EntropyRegularization` | `EuclideanNormRegularization` | `Fierce` |
| `GANFeatureMatching` | `GMVAE` | `LabelSmoothing` | `LayerScale` |
| `LCC` | `LVR` | `ManifoldMixup` | `OffDiagonalOrthogonalRegularization` |
| `OrthogonalRegularization` | `PathLengthRegularization` | `PGM` | `R1Regularization` |
| `Rome` | `SCN` | `ShakeShakeRegularization` | `SRN` |
| `StochasticDepth` | `STTP` | `SVDParameterization` | `TargetPolicySmoothing` |
| `TemporalActivationRegularization` | `WeightsReset` | | |

!!! info "Implementation Details"
The implementation of regularization techniques can vary greatly. Some act like loss functions, others are modules to be applied in the `forward` pass, and some might be implemented as training callbacks. Always refer to the specific header file in `<xtorch/regulariztions/>` for detailed usage instructions and available options.
