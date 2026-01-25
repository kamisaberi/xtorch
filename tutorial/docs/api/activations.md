# Activations

Activation functions are a critical component of neural networks, introducing non-linear properties that allow the network to learn complex patterns.

xTorch provides access to all standard activation functions from LibTorch and dramatically expands this collection with dozens of modern and experimental alternatives.

## Standard LibTorch Activations

All standard PyTorch activation functions are available directly through LibTorch. You can use them either as modules (e.g., `torch::nn::ReLU`) or as functional calls (e.g., `torch::nn::functional::relu`).

**Common Examples:**
- `torch::nn::ReLU` / `torch::relu`
- `torch::nn::Sigmoid` / `torch::sigmoid`
- `torch::nn::Tanh` / `torch::tanh`
- `torch::nn::Softmax` / `torch::softmax`
- `torch::nn::LeakyReLU` / `torch::leaky_relu`
- `torch::nn::GELU`

For a complete list and usage details, please refer to the [official PyTorch C++ documentation](https://pytorch.org/cppdocs/api/namespace_torch__nn.html).

---

## xTorch Extended Activations

In addition to the standard functions, xTorch includes a massive library of activation functions proposed in various research papers. This allows for easy experimentation with cutting-edge or specialized non-linearities without needing to implement them from scratch.

### Usage

All xTorch activations are implemented as `torch::nn::Module`s and can be found under the `xt::activations` namespace. They can be instantiated and used just like any standard `torch::nn` module.

```cpp
#include <xtorch/xtorch.h>

// Define a simple model
struct MyModel : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    // Instantiate an xTorch activation function
    xt::activations::Mish mish;

    MyModel() {
        fc1 = register_module("fc1", torch::nn::Linear(784, 128));
        // The activation is registered automatically by instantiation
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = fc1(x);
        // Apply the activation
        x = mish(x);
        x = fc2(x);
        return x;
    }
};
```

### Available Activations

Below is a comprehensive list of the activation functions available in the `xt::activations` module.

!!! note
Some activations may have constructor arguments for initialization parameters (e.g., `alpha` or `beta`). Refer to the corresponding header file in `<xtorch/activations/>` for specific details.

| | | | |
|---|---|---|---|
| `AGLU` | `AhaF` | `AMLines` | `Andhra` |
| `Aria` | `ASAF` | `ASU` | `CoLU` |
| `CosLU` | `CReLU` | `DeLU` | `DRA` |
| `ELiSH` | `ESwish` | `EvoNormS0` | `FEM` |
| `GCLU` | `GCU` | `GEGLU` | `GoLU` |
| `Gumbel` | `HardELiSH` | `HardSwish` | `HeLU` |
| `Hermite` | `KAF` | `KAN` | `LEAF` |
| `LinComb` | `Marcsinh` | `MarginReLU` | `Maxout` |
| `Mish` | `ModReLU` | `NailOr` | `NFN` |
| `Nipuna` | `NLSig` | `NormLinComb` | `PAU` |
| `Phish` | `PMish` | `Poly` | `Rational` |
| `ReGLU` | `ReLUN` | `RReLU` | `ScaledSoftSign` |
| `SERF` | `SeRLU` | `ShiftedSoftplus` | `ShiLU` |
| `SiLU` | `SIREN` | `Smish` | `SmoothStep` |
| `Splash` | `SquaredReLU` | `SRelu` | `StarReLU` |
| `SwiGELU` | `Swish` | `TAAF` | `TanhExp` |

