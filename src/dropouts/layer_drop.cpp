#include <dropouts/layer_drop.h>


// #include <torch/torch.h>
// #include <ostream> // For std::ostream
//
// struct LayerDropImpl : torch::nn::Module {
//     double p_drop_layer_; // Probability of dropping the layer this module gates.
//
//     LayerDropImpl(double p_drop_layer = 0.1) : p_drop_layer_(p_drop_layer) {
//         TORCH_CHECK(p_drop_layer_ >= 0.0 && p_drop_layer_ <= 1.0,
//                     "LayerDrop probability p_drop_layer must be between 0 and 1.");
//     }
//
//     // This forward method doesn't modify the input tensor directly.
//     // It returns a boolean indicating whether the layer associated with this
//     // LayerDrop instance should be "dropped" (skipped) for the current forward pass.
//     // The decision is made per forward call, typically affecting the entire batch uniformly
//     // for the specific layer it gates.
//     bool forward() { // Input tensor is not strictly needed for the decision but could be for other variants
//         if (!this->is_training() || p_drop_layer_ == 0.0) {
//             return false; // Never drop if not training or p_drop is 0
//         }
//         if (p_drop_layer_ == 1.0) {
//             return true; // Always drop if p_drop is 1
//         }
//
//         // Randomly decide whether to drop the layer
//         return torch::rand({1}).item<double>() < p_drop_layer_;
//     }
//
//     // An alternative forward if you prefer to pass the input and have the module
//     // return input if dropped, or a placeholder if kept (to be processed by actual layer).
//     // This example focuses on the boolean decision model.
//     /*
//     torch::Tensor forward(const torch::Tensor& input_to_potentially_droppable_layer) {
//         if (!this->is_training() || p_drop_layer_ == 0.0) {
//             // Process the layer (by returning something that signals 'process')
//             // Or, more simply, the calling code checks this LayerDrop and decides.
//             // For this version, let's assume it returns input if dropped.
//             return input_to_potentially_droppable_layer; // Placeholder if not dropping
//         }
//         if (p_drop_layer_ == 1.0) {
//             return input_to_potentially_droppable_layer; // Layer is dropped, effectively an identity
//         }
//
//         if (torch::rand({1}).item<double>() < p_drop_layer_) {
//             // Drop the layer: return the input itself, effectively making the layer an identity.
//             return input_to_potentially_droppable_layer;
//         } else {
//             // Don't drop the layer: here, we'd typically signal to the calling code
//             // to process the layer. For a self-contained module, this is tricky.
//             // The boolean return from the other `forward()` is cleaner for external control.
//             // This version is less common for LayerDrop's typical usage pattern.
//             // Let's stick to the boolean one for clarity of its role as a gate controller.
//             // If this `forward` was to return the actual processed output, it would need the layer.
//             return input_to_potentially_droppable_layer; // Placeholder, real processing happens outside
//         }
//     }
//     */
//
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "LayerDrop(p_drop_layer=" << p_drop_layer_ << ")";
//     }
// };
//
// TORCH_MODULE(LayerDrop); // Creates the LayerDrop module "class"
//
// /*
// // Example of how LayerDrop might be used in a model with multiple layers:
// // (This is for illustration and not part of LayerDropImpl itself)
//
// #include <vector>
// #include <iostream>
//
// // A dummy layer that just adds 1 to its input
// struct MySimpleLayer : torch::nn::Module {
//     MySimpleLayer() {}
//     torch::Tensor forward(const torch::Tensor& x) {
//         return x + 1.0;
//     }
// };
// TORCH_MODULE(MySimpleLayer);
//
//
// struct ModelWithLayerDrop : torch::nn::Module {
//     std::vector<MySimpleLayer> layers_;
//     std::vector<LayerDrop> layer_drop_modules_; // One LayerDrop decider per actual layer
//
//     ModelWithLayerDrop(int num_layers = 4, double layer_drop_rate = 0.2) {
//         layers_.reserve(num_layers);
//         layer_drop_modules_.reserve(num_layers);
//         for (int i = 0; i < num_layers; ++i) {
//             layers_.push_back(register_module("layer_" + std::to_string(i), MySimpleLayer()));
//             // Each layer gets its own LayerDrop decider module
//             layer_drop_modules_.push_back(
//                 register_module("layer_drop_" + std::to_string(i), LayerDrop(layer_drop_rate))
//             );
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         int layers_processed_count = 0;
//         for (size_t i = 0; i < layers_.size(); ++i) {
//             bool drop_this_layer = false;
//             if (this->is_training()) { // LayerDrop decision only relevant in training
//                 drop_this_layer = layer_drop_modules_[i]->forward();
//             }
//
//             if (drop_this_layer) {
//                 // Layer is "dropped": skip its computation (effectively an identity function)
//                 // std::cout << "Dropping layer " << i << std::endl; // For debugging
//             } else {
//                 // Layer is processed
//                 x = layers_[i]->forward(x);
//                 layers_processed_count++;
//                 // std::cout << "Processing layer " << i << std::endl; // For debugging
//             }
//         }
//         // std::cout << "Total layers processed: " << layers_processed_count << std::endl;
//         return x;
//     }
// };
// TORCH_MODULE(ModelWithLayerDrop);
//
//
// void run_layer_drop_example() {
//     torch::manual_seed(0); // For reproducible results
//
//     double drop_rate = 0.5; // 50% chance of dropping each layer
//     int num_actual_layers = 6;
//
//     ModelWithLayerDrop model(num_actual_layers, drop_rate);
//     std::cout << "Model with LayerDrop modules: " << model << std::endl;
//
//     torch::Tensor input_tensor = torch::zeros({1, 1}); // Simple input
//
//     // --- Training mode ---
//     model->train(); // Set the model (and its LayerDrop submodules) to training mode
//     std::cout << "\n--- Training Mode ---" << std::endl;
//     for (int i = 0; i < 5; ++i) { // Run a few times to see different drop patterns
//         torch::Tensor output_train = model->forward(input_tensor);
//         std::cout << "Run " << i << ": Output (train) = " << output_train.item<float>()
//                   << " (Input was 0, each processed layer adds 1)" << std::endl;
//         // The output value will indicate how many layers were *not* dropped.
//     }
//
//     // --- Evaluation mode ---
//     model->eval(); // Set to evaluation mode
//     torch::Tensor output_eval = model->forward(input_tensor);
//     std::cout << "\n--- Evaluation Mode ---" << std::endl;
//     std::cout << "Output (eval) = " << output_eval.item<float>() << std::endl;
//     // Expected: All layers should be processed, so output should be num_actual_layers (0 + 1*6 = 6).
//     TORCH_CHECK(output_eval.item<float>() == static_cast<float>(num_actual_layers), "LayerDrop eval output mismatch!");
//
//
//     // --- Test with p_drop_layer = 0.0 (no layer dropping) ---
//     ModelWithLayerDrop no_drop_model(num_actual_layers, 0.0);
//     no_drop_model->train();
//     torch::Tensor output_no_drop_train = no_drop_model->forward(input_tensor);
//     std::cout << "\nOutput (train, p_drop_layer=0.0) = " << output_no_drop_train.item<float>() << std::endl;
//     TORCH_CHECK(output_no_drop_train.item<float>() == static_cast<float>(num_actual_layers), "LayerDrop p_drop=0.0 output mismatch!");
//
//
//     // --- Test with p_drop_layer = 1.0 (always drop all layers) ---
//     ModelWithLayerDrop full_drop_model(num_actual_layers, 1.0);
//     full_drop_model->train();
//     torch::Tensor output_full_drop_train = full_drop_model->forward(input_tensor);
//     std::cout << "Output (train, p_drop_layer=1.0) = " << output_full_drop_train.item<float>() << std::endl;
//     // Expected: All layers dropped, output should be same as input (0).
//     TORCH_CHECK(output_full_drop_train.item<float>() == input_tensor.item<float>(), "LayerDrop p_drop=1.0 output mismatch!");
//
// }
//
// // int main() {
// //    run_layer_drop_example();
// //    return 0;
// // }
// */


namespace xt::dropouts
{
    LayerDrop::LayerDrop(double p_drop_layer ) : p_drop_layer_(p_drop_layer)
    {
        TORCH_CHECK(p_drop_layer_ >= 0.0 && p_drop_layer_ <= 1.0,
                    "LayerDrop probability p_drop_layer must be between 0 and 1.");
    }

    auto LayerDrop::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto input_to_potentially_droppable_layer = std::any_cast<torch::Tensor>(tensors_[0]);

        if (!this->is_training() || p_drop_layer_ == 0.0)
        {
            // Process the layer (by returning something that signals 'process')
            // Or, more simply, the calling code checks this LayerDrop and decides.
            // For this version, let's assume it returns input if dropped.
            return input_to_potentially_droppable_layer; // Placeholder if not dropping
        }
        if (p_drop_layer_ == 1.0)
        {
            return input_to_potentially_droppable_layer; // Layer is dropped, effectively an identity
        }

        if (torch::rand({1}).item<double>() < p_drop_layer_)
        {
            // Drop the layer: return the input itself, effectively making the layer an identity.
            return input_to_potentially_droppable_layer;
        }
        else
        {
            // Don't drop the layer: here, we'd typically signal to the calling code
            // to process the layer. For a self-contained module, this is tricky.
            // The boolean return from the other `forward()` is cleaner for external control.
            // This version is less common for LayerDrop's typical usage pattern.
            // Let's stick to the boolean one for clarity of its role as a gate controller.
            // If this `forward` was to return the actual processed output, it would need the layer.
            return input_to_potentially_droppable_layer; // Placeholder, real processing happens outside
        }
    }
}
