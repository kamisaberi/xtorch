# Optimizers

An optimizer is an algorithm that adapts the neural network's attributes, such as weights and learning rates, to minimize the loss function. The choice of optimizer can have a significant impact on training speed and final model performance.

## Standard LibTorch Optimizers

LibTorch provides a robust set of the most common and well-established optimization algorithms, which are suitable for a wide range of tasks.

**Common Examples:**
- `torch::optim::SGD`
- `torch::optim::Adam`
- `torch::optim::RMSprop`
- `torch::optim::Adagrad`

These optimizers are used by passing the model's parameters and an `Options` struct to their constructor. For a complete guide, please refer to the [official PyTorch C++ optimizer documentation](https://pytorch.org/cppdocs/api/namespace_torch__optim.html).

---

## xTorch Extended Optimizers

The field of optimization is an active area of research, with new and improved algorithms being published regularly. To empower developers and researchers to leverage these advancements, xTorch includes a massive library of modern and specialized optimizers.

This allows you to easily replace `Adam` with variants like `RAdam`, `AdamW`, `AdaBelief`, or `LAMB` to see if they improve your model's convergence or generalization.

### Usage

All xTorch optimizers are located in the `xt::optimizations` namespace. They are designed to be a drop-in replacement for standard `torch::optim` optimizers. You construct them in the same way: by providing the model's parameters and an options struct.

They integrate perfectly with the `xt::Trainer`.

```cpp
#include <xtorch/xtorch.hh>

int main() {
    // 1. Assume 'model' is an initialized torch::nn::Module
    xt::models::LeNet5 model(10);
    model.to(torch::kCPU);

    // 2. Instantiate an xTorch optimizer
    // It takes the model parameters and an Options struct, just like a standard optimizer.
    xt::optimizations::RAdam optimizer(
        model.parameters(),
        xt::optimizations::RAdamOptions(1e-3) // Learning rate of 0.001
    );

    // 3. Integrate with the xt::Trainer
    xt::Trainer trainer;
    trainer.set_max_epochs(10)
           .set_optimizer(optimizer) // Pass the xTorch optimizer to the trainer
           .set_loss_fn(torch::nll_loss);

    // The trainer will now use RAdam to update the model's weights.
    // trainer.fit(model, data_loader, nullptr, torch::kCPU);

    std::cout << "Trainer configured with RAdam optimizer." << std::endl;
}
```

### Available Optimizers

Below is the comprehensive list of optimizers available in the `xt::optimizations` module.

| | | | |
|---|---|---|---|
| `OneBitAdam` | `OneBitLamb` | `AdaBelief` | `AdaBound` |
| `Adafactor` | `AdaFisher` | `AdaHessian` | `AdaMax` |
| `AdamMini` | `AdaMod` | `AdamW` | `AdaShift` |
| `AdaSmooth` | `AdaSqrt` | `ADOPT` | `AggMo` |
| `AMSBound` | `AMSGrad` | `AO` | `Apollo` |
| `Atmo` | `DeepEnsembles` | `DemonAdam` | `DemonCM` |
| `Demon` | `DFA` | `DiagAdaFisher` | `DistributedShampoo` |
| `DSPT` | `ECO` | `FA` | `FASFA` |
| `FATA` | `ForwardGradient` | `GCANS` | `GradientCheckpointing` |
| `GradientSparsification` | `Gravity` | `HGS` | `Info` |
| `KP` | `LAMB` | `LARS` | `LocalSGD` |
| `Lookahead` | `MadGrad` | `MAS` | `MPSO` |
| `Nadam` | `NTASGD` | `PLO` | `PO` |
| `PowerPropagation` | `PowerSGD` | `QHAdam` | `QHM` |
| `RAdam` | `SLamb` | `SM3` | `SMA` |
| `SRMM` | `StochasticWeightAveraging` | `YellowFin` |

!!! info "Constructor Options"
Each optimizer has its own set of hyperparameters (e.g., `lr`, `betas`, `eps`, `weight_decay`). These are configured via a dedicated `Options` struct passed to the constructor. Please refer to the specific header file in `<xtorch/optimizations/>` for details on the available settings for each optimizer.
