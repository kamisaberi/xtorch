#include "include/dropouts/sensor_dropout.h"



// #include <torch/torch.h>
// #include <vector>
// #include <numeric>   // For std::iota, std::accumulate
// #include <algorithm> // For std::shuffle
// #include <random>    // For std::mt19937, std::random_device
// #include <ostream>   // For std::ostream
//
// struct SensorDropoutImpl : torch::nn::Module {
//     double p_drop_sensor_; // Probability of dropping any given sensor
//     std::vector<int64_t> sensor_splits_; // Number of features for each sensor, e.g., {10, 5, 15} for 3 sensors
//     int64_t total_features_;
//     double epsilon_ = 1e-7;
//
//     SensorDropoutImpl(const std::vector<int64_t>& sensor_splits, double p_drop_sensor = 0.1)
//         : p_drop_sensor_(p_drop_sensor), sensor_splits_(sensor_splits) {
//         TORCH_CHECK(p_drop_sensor_ >= 0.0 && p_drop_sensor_ <= 1.0,
//                     "p_drop_sensor must be between 0 and 1.");
//         TORCH_CHECK(!sensor_splits_.empty(), "sensor_splits cannot be empty.");
//         for (int64_t split_size : sensor_splits_) {
//             TORCH_CHECK(split_size > 0, "Each sensor split size must be positive.");
//         }
//         total_features_ = std::accumulate(sensor_splits_.begin(), sensor_splits_.end(), 0LL);
//     }
//
//     // Input x is expected to be (Batch, TotalFeatures) or (TotalFeatures)
//     // where TotalFeatures is the sum of sensor_splits_.
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training() || p_drop_sensor_ == 0.0 || sensor_splits_.empty()) {
//             return input;
//         }
//
//         TORCH_CHECK(input.size(-1) == total_features_,
//                     "Input's last dimension size (", input.size(-1),
//                     ") does not match total_features_ from sensor_splits (", total_features_, ").");
//
//         int num_sensors = sensor_splits_.size();
//         if (num_sensors == 0) return input;
//
//         // Create a per-sensor keep/drop decision mask
//         // This mask is (num_sensors), 1 if sensor is kept, 0 if dropped.
//         torch::Tensor sensor_decision_mask_1d = torch::bernoulli(
//             torch::full({num_sensors}, 1.0 - p_drop_sensor_, input.options())
//         ).to(input.dtype());
//
//         // If all sensors are decided to be dropped (unlikely but possible if p_drop_sensor_ is high)
//         // and p_drop_sensor_ < 1.0, we might want to ensure at least one sensor is kept.
//         // For simplicity now, we allow all to be dropped if p_drop_sensor_ is high enough.
//         // If p_drop_sensor_ == 1.0, all sensors are dropped.
//         if (p_drop_sensor_ == 1.0) {
//             return torch::zeros_like(input);
//         }
//
//         // Construct the full feature mask based on sensor_decision_mask_1d
//         std::vector<torch::Tensor> feature_mask_parts;
//         feature_mask_parts.reserve(num_sensors);
//         int64_t num_kept_sensors = 0;
//         int64_t num_kept_features = 0;
//
//         for (int i = 0; i < num_sensors; ++i) {
//             bool keep_this_sensor = sensor_decision_mask_1d[i].item<double>() > 0.5;
//             int64_t current_sensor_features = sensor_splits_[i];
//             if (keep_this_sensor) {
//                 feature_mask_parts.push_back(torch::ones({current_sensor_features}, input.options().dtype(input.dtype())));
//                 num_kept_sensors++;
//                 num_kept_features += current_sensor_features;
//             } else {
//                 feature_mask_parts.push_back(torch::zeros({current_sensor_features}, input.options().dtype(input.dtype())));
//             }
//         }
//
//         // If all sensors happened to be dropped by chance (and p_drop_sensor < 1.0),
//         // to prevent division by zero or empty output, one could force keeping one sensor.
//         // For this version, if num_kept_features is 0, we'll return zeros.
//         // A more robust way would be to ensure at least one sensor is kept unless p_drop_sensor_ == 1.0.
//         // This can be done by re-sampling sensor_decision_mask_1d if it's all zeros,
//         // or by adjusting the scaling factor.
//         if (num_kept_features == 0 && p_drop_sensor_ < 1.0) {
//             // This situation means all sensors were dropped by chance.
//             // If the goal is to always have some signal, one might force-keep a random sensor.
//             // Or, accept that this pass has no sensor data.
//             // For scaling, this means scale factor would be infinite. Let's return zeros.
//             return torch::zeros_like(input);
//         }
//
//
//         torch::Tensor feature_mask = torch::cat(feature_mask_parts, 0); // Shape (TotalFeatures)
//
//         // Reshape feature_mask to be broadcastable with input
//         // If input is (Batch, TotalFeatures), mask needs to be (1, TotalFeatures) or (TotalFeatures)
//         // If input is (TotalFeatures), mask is fine.
//         if (input.dim() > 1 && feature_mask.dim() == 1) { // e.g. input (B,F), mask (F)
//              // feature_mask is already (TotalFeatures), broadcasting rules will handle it
//         }
//
//         // Scaling: The paper "DropBlock" and others scale by (total_area / kept_area).
//         // Here, we scale by (total_num_features / num_kept_features) IF we only considered
//         // the features of active sensors.
//         // A simpler scaling is to scale by 1 / (1 - effective_feature_drop_rate).
//         // The effective feature drop rate is not simply p_drop_sensor_.
//         // Let's scale by the proportion of sensors kept, assuming features are somewhat evenly important.
//         // Or, more accurately, by proportion of features kept.
//         double scale_factor;
//         if (num_kept_features > 0) {
//             scale_factor = static_cast<double>(total_features_) / (static_cast<double>(num_kept_features) + epsilon_);
//         } else { // All features dropped
//             scale_factor = 0; // Effectively, output is zero. This happens if p_drop_sensor_ = 1.0 or by chance.
//         }
//
//         return (input * feature_mask) * scale_factor;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "SensorDropout(p_drop_sensor=" << p_drop_sensor_
//                << ", num_sensors=" << sensor_splits_.size()
//                << ", total_features=" << total_features_ << ")";
//     }
// };
//
// TORCH_MODULE(SensorDropout);
//
// /*
// // Example of how to use the SensorDropout module:
// #include <iostream>
//
// void run_sensor_dropout_example() {
//     torch::manual_seed(0);
//
//     std::vector<int64_t> sensor_config = {3, 2, 5}; // Sensor 1: 3 feats, Sensor 2: 2 feats, Sensor 3: 5 feats
//                                                     // Total features = 10
//     double prob_drop_one_sensor = 0.4;
//
//     SensorDropout sensor_dropout_module(sensor_config, prob_drop_one_sensor);
//     std::cout << "SensorDropout Module: " << sensor_dropout_module << std::endl;
//
//     // Input: Batch of 2 samples, each with 10 features
//     torch::Tensor input_tensor = torch::arange(1, 21, torch::kFloat).reshape({2, 10});
//     // Sample 0: [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
//     //            <--S1--> <--S2--> <---- S3 ---->
//     // Sample 1: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
//
//     std::cout << "Input Tensor (Batch=2, TotalFeatures=10):\n" << input_tensor << std::endl;
//
//     // --- Training mode ---
//     sensor_dropout_module->train();
//     std::cout << "\n--- Training Mode ---" << std::endl;
//     for (int i = 0; i < 5; ++i) { // Run a few times to see different sensor drop patterns
//         torch::Tensor output_train = sensor_dropout_module->forward(input_tensor);
//         std::cout << "Run " << i << ": Output (train):\n" << output_train << std::endl;
//         // Check which sensors were effectively dropped by looking for all zeros in their feature segments.
//         // Example: if sensor 2 (features 3,4 for 0-indexed) is dropped for sample 0,
//         // output_train[0][3] and output_train[0][4] should be 0.
//         // Kept sensors' features will be scaled.
//     }
//
//     // --- Evaluation mode ---
//     sensor_dropout_module->eval();
//     torch::Tensor output_eval = sensor_dropout_module->forward(input_tensor);
//     std::cout << "\n--- Evaluation Mode ---" << std::endl;
//     std::cout << "Output (eval):\n" << output_eval << std::endl;
//     TORCH_CHECK(torch::allclose(input_tensor, output_eval), "SensorDropout eval output mismatch!");
//
//
//     // --- Test with p_drop_sensor = 0.0 (no sensor dropping) ---
//     SensorDropout no_drop_module(sensor_config, 0.0);
//     no_drop_module->train();
//     torch::Tensor output_no_drop_train = no_drop_module->forward(input_tensor);
//     std::cout << "\nOutput (train, p_drop_sensor=0.0):\n" << output_no_drop_train << std::endl;
//     TORCH_CHECK(torch::allclose(input_tensor, output_no_drop_train), "SensorDropout p_drop=0.0 output mismatch!");
//     // Scale factor should be 1.0 (10/10)
//
//
//     // --- Test with p_drop_sensor = 1.0 (drop all sensors) ---
//     SensorDropout full_drop_module(sensor_config, 1.0);
//     full_drop_module->train();
//     torch::Tensor output_full_drop_train = full_drop_module->forward(input_tensor);
//     std::cout << "\nOutput (train, p_drop_sensor=1.0):\n" << output_full_drop_train << std::endl;
//     TORCH_CHECK(torch::allclose(torch::zeros_like(input_tensor), output_full_drop_train), "SensorDropout p_drop=1.0 output mismatch!");
//     // Scale factor should be 0, output is all zeros.
// }
//
// // int main() {
// //    run_sensor_dropout_example();
// //    return 0;
// // }
// */
//



namespace xt::dropouts
{
    torch::Tensor sensor_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto SensorDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::sensor_dropout(torch::zeros(10));
    }
}
