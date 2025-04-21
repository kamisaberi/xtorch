# xtorch: Complete C++ Libtorch Extension

```cpp
// File: xtorch_quickstart.cpp
// One-file demonstration of all major features
#include <xtorch/xtensor.hpp>
#include <xtorch/xnn.hpp>
#include <xtorch/xoptim.hpp>
#include <xtorch/xutils.hpp>

// 1. MEMORY-OPTIMIZED TENSORS
void tensor_demo() {
    auto x = xtorch::XTensor::arange(9).reshape({3,3});
    auto y = xtorch::XTensor::randn({3,3});
    
    // Fused operations
    auto z = x.fused_mm(y).inplace_sigmoid();
    
    // Smart casting
    z.smart_cast(torch::kHalf); 
}

// 2. NEURAL NETWORK MODULE
struct XModel : torch::nn::Module {
    xtorch::XLinear fc{nullptr};
    xtorch::XGELU act{nullptr};
    
    XModel(int in, int out) : 
        fc(register_module("fc", xtorch::XLinear(in, out))),
        act(register_module("act", xtorch::XGELU())) {}
    
    torch::Tensor forward(torch::Tensor x) {
        return act(fc(x));
    }
};

// 3. TRAINING PIPELINE
void train() {
    // Components
    auto model = std::make_shared<XModel>(784, 10);
    xtorch::XAdam optimizer(model->parameters());
    auto loader = xtorch::FastMNISTLoader("./data");
    
    // Memory context
    xtorch::mem::OptimizationContext ctx;
    ctx.enable_inplace = true;

    // Training loop
    for(auto& batch : loader) {
        optimizer.zero_grad();
        auto loss = torch::nll_loss(
            model->forward(batch.data), 
            batch.target
        );
        loss.backward();
        optimizer.step(ctx);
        xtorch::mem::auto_release_cache();
    }
}

// 4. DEPLOYMENT
void export_model(std::shared_ptr<XModel> model) {
    // JIT tracing
    auto traced = torch::jit::trace(model, torch::ones({1,784}));
    traced->save("model.pt");
    
    // ONNX export
    xtorch::export::to_onnx(model, "model.onnx", 
        {torch::ones({1,784})}, {"input"}
    );
}

// MAIN DEMO
int main() {
    tensor_demo();
    
    auto model = std::make_shared<XModel>(784, 10);
    train();
    export_model(model);
    
    return 0;
}