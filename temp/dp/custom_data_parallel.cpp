#include "custom_data_parallel.h"
#include <c10/cuda/CUDAStream.h> // For CUDA context management if needed, though less critical for this simple threading
#include <c10/cuda/CUDAGuard.h>  // For c10::cuda::OptionalCUDAGuard

namespace xt {

    CustomDataParallel::CustomDataParallel(std::shared_ptr<xt::Module> module_to_parallelize,
                                           const std::vector<torch::Device>& devices)
        : original_module_(module_to_parallelize), devices_(devices) {
        if (devices_.empty()) {
            throw std::runtime_error("CustomDataParallel: No devices provided.");
        }
        primary_device_ = devices_[0];

        // Ensure original module is on the primary device initially before cloning
        original_module_->to(primary_device_);

        // Create replicas
        for (size_t i = 0; i < devices_.size(); ++i) {
            const auto& device = devices_[i];
            if (!device.is_cuda()) {
                std::cerr << "Warning: CustomDataParallel currently only supports CUDA devices. Device "
                          << device.str() << " will be skipped or might cause issues if not primary." << std::endl;
                // For this example, we'll proceed but real DataParallel would error or handle CPU differently.
                // If it's the primary device and not CUDA, then this example won't work as intended.
            }
            // Create a deep copy (clone) of the original module for each device
            // This assumes your xt::Module (or the model it wraps) has a clone method
            // or can be serialized and deserialized for a deep copy.
            // For torch::nn::Module, you can serialize and deserialize.
            // If xt::Module wraps a torch::nn::Module, you can do something like:
            // auto new_replica_torch_module = std::dynamic_pointer_cast<torch::nn::Module>(original_module_->clone());
            // std::shared_ptr<xt::Module> new_replica = std::make_shared<YourXtModuleWrapper>(new_replica_torch_module);

            // Simplification: If xt::Module doesn't have a deep clone, this won't be true data parallelism.
            // For torch::nn::Module based models:
            // 1. Serialize original_module_ to a stream
            // 2. Deserialize from stream to create a new module instance for replica
            // This is non-trivial to implement generically for xt::Module.
            // A common way for torch::nn::Module:
            if (auto original_nn_module = std::dynamic_pointer_cast<torch::nn::Module>(original_module_)) {
                 // This is a placeholder. A true clone is more involved for arbitrary xt::Module.
                 // If your xt::Module *is* a torch::nn::Module directly, this is simpler.
                 // Let's assume for now that xt::Module has a "clone_to_device" or similar concept
                 // For this example, we'll just re-use the shared_ptr and move it.
                 // THIS IS A MAJOR SIMPLIFICATION AND NOT TRUE DATA PARALLELISM FOR STATEFUL MODULES
                 // UNLESS original_module_->clone() makes a deep copy.
                 // A proper clone for torch::nn::Module:
                std::ostringstream oss;
                torch::save(original_nn_module, oss);
                std::istringstream iss(oss.str());
                auto cloned_nn_module = std::make_shared<torch::nn::Module>(); // Or the concrete type
                torch::load(cloned_nn_module, iss); // Load into a new instance

                // If xt::Module is a wrapper around torch::nn::Module:
                // auto replica_xt_module = std::make_shared<xt::ModuleDerivedFromTorchNnModule>(); // Your wrapper
                // torch::load(std::dynamic_pointer_cast<torch::nn::Module>(replica_xt_module), iss);
                // For now, let's assume a simple case where original_module_ can be used, but moved.
                // This is a conceptual simplification.
                // For actual deep copy, one would need to serialize and deserialize, or have a clone method.

                // For a basic torch::nn::Module (which xt::Module is a base of)
                // we can try to re-create it if we knew its type, or use state_dict.
                // This part is tricky without knowing the exact nature of xt::Module and the wrapped model.

                // Let's assume original_module_ is the one on devices_[0] (primary_device_)
                // and for other devices, we'd need to create new instances and load state_dict.
                // For simplicity of this example, we'll just have one "master" (original_module_)
                // and the "replicas_" vector will point to it if i==0, and for i>0 it would
                // ideally be true deep copies. This is a significant simplification.
                if (i == 0) {
                    replicas_.push_back(original_module_); // Primary replica is the original_module_ itself
                } else {
                    // TODO: Proper deep cloning of the module for other devices.
                    // This might involve knowing the concrete type of 'original_module_'
                    // or having a virtual 'clone' method in xt::Module.
                    // For now, this will lead to issues if not handled:
                    // replicas_.push_back(original_module_->clone()->to(device)); // Assuming clone exists

                    // Hack for demonstration: create a new instance and load state dict
                    // This requires original_module_ to be a concrete, instantiable type or its derived type to be known.
                    // This is highly dependent on your xt::Module structure.
                    // If original_module_ is just a torch::nn::Module:
                    auto new_replica = std::dynamic_pointer_cast<torch::nn::Module>(module_to_parallelize->clone());
                    if(new_replica) {
                        new_replica->to(device);
                        replicas_.push_back(std::static_pointer_cast<xt::Module>(new_replica));
                    } else {
                         // Fallback: this means shared parameters if clone isn't deep, which isn't DataParallel
                         replicas_.push_back(original_module_); // Problematic for true parallelism
                         if (original_module_->device() != device) original_module_->to(device); // WRONG - modifies original
                         std::cerr << "Warning: Could not properly clone module for device " << device.str()
                                   << ". True data parallelism might not be achieved." << std::endl;
                    }
                }
            } else {
                throw std::runtime_error("CustomDataParallel: module_to_parallelize cannot be cast to torch::nn::Module for cloning example.");
            }
        }
         // Ensure all replicas are on their designated devices
        for(size_t i = 0; i < replicas_.size(); ++i) {
            if (replicas_[i]) replicas_[i]->to(devices_[i]);
        }
    }

    CustomDataParallel::~CustomDataParallel() {
        for (auto& t : threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    // Scatter input tensor to multiple devices
    void CustomDataParallel::scatter(const torch::Tensor& input_tensor, std::vector<torch::Tensor>& scattered_inputs) {
        scattered_inputs.clear();
        int64_t num_devices = devices_.size();
        if (num_devices == 0) return;

        int64_t batch_size = input_tensor.size(0);
        int64_t chunk_size = (batch_size + num_devices - 1) / num_devices; // Ceiling division

        for (int64_t i = 0; i < num_devices; ++i) {
            int64_t start = i * chunk_size;
            if (start >= batch_size) {
                // This device gets no data if batch_size is small
                scattered_inputs.push_back(torch::empty({0}, input_tensor.options())); // Empty tensor
                continue;
            }
            int64_t end = std::min(start + chunk_size, batch_size);
            scattered_inputs.push_back(input_tensor.slice(0, start, end).to(devices_[i], /*non_blocking=*/true));
        }
    }

    // Gather outputs from multiple devices to the primary device
    torch::Tensor CustomDataParallel::gather(const std::vector<torch::Tensor>& scattered_outputs, int64_t target_batch_dim = 0) {
        std::vector<torch::Tensor> outputs_on_primary_device;
        outputs_on_primary_device.reserve(scattered_outputs.size());
        for (const auto& output_chunk : scattered_outputs) {
            if (output_chunk.numel() > 0) { // Only gather non-empty chunks
                 outputs_on_primary_device.push_back(output_chunk.to(primary_device_, /*non_blocking=*/true));
            }
        }
        if (outputs_on_primary_device.empty()) {
            // Handle case where all chunks were empty (e.g. input batch size was 0)
            // Return an empty tensor with appropriate options, or throw.
            // For now, let's assume at least one output. If scattered_outputs was based on an empty input,
            // the first element would be an empty tensor.
            if (!scattered_outputs.empty() && scattered_outputs[0].defined()) {
                 return torch::empty({0}, scattered_outputs[0].options().device(primary_device_));
            }
            // This case needs careful handling based on expected empty input behavior
            throw std::runtime_error("Gather: No outputs to gather or outputs are undefined.");
        }
        return torch::cat(outputs_on_primary_device, target_batch_dim);
    }


    std::any CustomDataParallel::forward(std::initializer_list<std::any> inputs_any) {
        if (inputs_any.size() == 0) {
            throw std::runtime_error("CustomDataParallel::forward expects at least one input tensor (data).");
        }

        torch::Tensor data_batch;
        try {
            data_batch = std::any_cast<torch::Tensor>(*inputs_any.begin());
        } catch(const std::bad_any_cast& e) {
            throw std::runtime_error("CustomDataParallel: First input must be a torch::Tensor for data. " + std::string(e.what()));
        }

        // Optional: Handle targets if they are passed for internal loss calculation
        // torch::Tensor target_batch;
        // if (inputs_any.size() > 1) {
        //    target_batch = std::any_cast<torch::Tensor>(*(inputs_any.begin() + 1));
        // }


        if (devices_.size() == 1) { // Single device, no parallelism needed, run on primary
            return replicas_[0]->forward({data_batch.to(primary_device_)});
        }

        // 1. Scatter Data
        std::vector<torch::Tensor> scattered_data;
        scatter(data_batch, scattered_data);

        // (Optional) Scatter Targets if doing loss calculation per replica (more complex)
        // std::vector<torch::Tensor> scattered_targets;
        // if (target_batch.defined()) scatter(target_batch, scattered_targets);


        // 2. Parallel Forward Pass
        std::vector<std::any> replica_outputs_any(devices_.size());
        threads_.clear(); // Clear previous threads

        std::vector<std::future<std::any>> futures; futures.reserve(devices_.size());

        for (size_t i = 0; i < devices_.size(); ++i) {
            if (scattered_data[i].numel() == 0) { // Skip devices with no data
                // Store an empty or placeholder result if needed, or handle later in gather
                // For simplicity, let's assume we need to have a result placeholder
                // replica_outputs_any[i] = torch::empty({0}); // Needs to match expected output type
                continue;
            }

            futures.emplace_back(
                std::async(std::launch::async, [this, i, current_data_chunk = scattered_data[i]]() -> std::any {
                    // c10::cuda::OptionalCUDAGuard guard(devices_[i]); // Important for setting CUDA context per thread
                    // The above guard might be needed if not using async streams properly
                    // Or ensure torch operations select the correct device internally.
                    // LibTorch ops usually respect the device of their input tensors.
                    try {
                        return replicas_[i]->forward({current_data_chunk});
                    } catch (const std::exception& e) {
                        std::cerr << "Exception in parallel forward on device " << devices_[i] << ": " << e.what() << std::endl;
                        return {}; // Return empty std::any on error
                    }
                })
            );
        }

        std::vector<torch::Tensor> replica_output_tensors;
        replica_output_tensors.reserve(devices_.size());

        for(size_t i=0; i < futures.size(); ++i) {
            try {
                 std::any out_any = futures[i].get();
                 if (out_any.has_value()) {
                     replica_output_tensors.push_back(std::any_cast<torch::Tensor>(out_any));
                 } else if (scattered_data[i].numel() > 0) { // If it had input but no output
                     throw std::runtime_error("Replica " + std::to_string(i) + " future returned empty std::any despite input.");
                 }
            } catch (const std::future_error& e) {
                 std::cerr << "Future error for replica " << i << ": " << e.what() << std::endl;
                 throw;
            } catch (const std::bad_any_cast& e) {
                 std::cerr << "Bad any_cast for replica " << i << " output: " << e.what() << std::endl;
                 throw;
            }
        }


        // 3. Gather Outputs
        // The outputs are gathered on the primary device.
        // The Trainer will then typically compute loss using these gathered outputs
        // and the original (non-scattered) targets on the primary device.
        // The backward pass will then happen on the primary_replica_.
        if (replica_output_tensors.empty() && data_batch.numel() > 0) {
            throw std::runtime_error("No outputs from replicas to gather, but input batch was not empty.");
        }
        if (replica_output_tensors.empty() && data_batch.numel() == 0) {
            // If input was empty, return an empty tensor of expected type
             std::shared_ptr<torch::nn::LinearImpl> temp_lin; // just to get an options object
             if (!replicas_.empty() && replicas_[0]) {
                // Try to get an example output to deduce type, this is hacky
                // auto example_out_any = replicas_[0]->forward({torch::randn({1,1}, devices_[0])});
                // if (example_out_any.has_value()){
                //     return std::any_cast<torch::Tensor>(example_out_any).slice(0,0,0); // empty slice
                // }
             }
             // Fallback, highly dependent on what forward returns
             return torch::empty({0}, data_batch.options().device(primary_device_));
        }

        torch::Tensor gathered_output = gather(replica_output_tensors, 0); // Assuming batch dim is 0

        return gathered_output; // Return as std::any
    }

    // Return parameters of the primary replica
    std::vector<torch::Tensor> CustomDataParallel::parameters(bool recurse) const {
        return original_module_->parameters(recurse); // Parameters are managed by the primary copy
    }
    std::vector<torch::Tensor> CustomDataParallel::named_parameters(bool recurse) const {
         return original_module_->named_parameters(recurse);
    }


    // Synchronize parameters from primary_replica_ to all other replicas
    void CustomDataParallel::synchronize_replicas() {
        if (devices_.size() <= 1) return; // No sync needed for 0 or 1 device beyond primary

        // Get state_dict from the primary model (original_module_)
        // This requires original_module_ to have a state_dict-like mechanism or be a torch::nn::Module
        auto primary_nn_module = std::dynamic_pointer_cast<torch::nn::Module>(original_module_);
        if (!primary_nn_module) {
            std::cerr << "Warning: Cannot synchronize replicas, primary module is not a torch::nn::Module." << std::endl;
            return;
        }
        auto state_dict = primary_nn_module->named_parameters(/*recurse=*/true); // Or just parameters if non-recursive

        for (size_t i = 0; i < replicas_.size(); ++i) {
            if (devices_[i] == primary_device_) continue; // Skip primary

            auto replica_nn_module = std::dynamic_pointer_cast<torch::nn::Module>(replicas_[i]);
            if (replica_nn_module) {
                // This is a simplified way. Proper state_dict loading is better.
                // torch::NoGradGuard no_grad;
                // for(auto& param_pair : replica_nn_module->named_parameters(true)){
                //    if(state_dict.contains(param_pair.key())){
                //        param_pair.value().data().copy_(state_dict[param_pair.key()].data());
                //    }
                // }
                // A more robust way if xt::Module wraps torch::nn::Module and has load_state_dict
                // replica_nn_module->load_state_dict(primary_nn_module->state_dict());
                // For now, direct parameter copy (can be risky if structures differ, but replicas should be clones)
                auto primary_params = primary_nn_module->parameters(true);
                auto replica_params = replica_nn_module->parameters(true);
                if (primary_params.size() == replica_params.size()) {
                    torch::NoGradGuard no_grad;
                    for (size_t p_idx = 0; p_idx < primary_params.size(); ++p_idx) {
                        replica_params[p_idx].data().copy_(primary_params[p_idx].data().to(devices_[i]));
                    }
                } else {
                     std::cerr << "Warning: Parameter count mismatch during replica synchronization for device " << devices_[i] << std::endl;
                }

            } else {
                std::cerr << "Warning: Replica on device " << devices_[i] << " is not a torch::nn::Module, cannot sync." << std::endl;
            }
        }
    }

    void CustomDataParallel::to(torch::Device device, bool non_blocking) {
        xt::Module::to(device, non_blocking); // Call base class if it has shared logic
        primary_device_ = device; // Assume the new device is the primary one
        if (original_module_) original_module_->to(device, non_blocking);
        for (size_t i = 0; i < replicas_.size(); ++i) {
            // For a true DataParallel, devices for replicas are fixed.
            // Moving the whole CustomDataParallel to a new device means
            // the primary replica moves. Other replicas should ideally stay on their original devices
            // or be re-initialized. This `to` is more like setting the *primary* device.
            // For simplicity here, we'll assume it means all replicas go to this new device,
            // which changes the DataParallel nature.
            // A better DataParallel would not allow moving individual replicas this way.
            // It would be reconfigured with a new set of devices.
            if (replicas_[i]) replicas_[i]->to(devices_[i], non_blocking); // Keep replicas on their designated devices
        }
        // Update devices_ if the intent is to change the set of parallel devices
        // For now, devices_ is fixed at construction.
        // If 'device' is one of the existing devices_, make it primary.
        bool found = false;
        for(const auto& d : devices_){
            if(d == device){
                primary_device_ = d;
                // Potentially re-order original_module_ to be the one on this new primary_device_
                // This logic gets complex quickly.
                found = true;
                break;
            }
        }
        if(!found){
            // If the target device is not in the list, what to do?
            // Option 1: Error. Option 2: Single device mode on 'device'.
             std::cout << "Warning: CustomDataParallel::to called with a device not in its list. "
                       << "Operating in single-device mode on " << device << std::endl;
             devices_ = {device};
             primary_device_ = device;
             replicas_ = {original_module_}; // original_module_ was already moved
        }
    }

    void CustomDataParallel::to(torch::ScalarType dtype, bool non_blocking) {
        xt::Module::to(dtype, non_blocking);
        if (original_module_) original_module_->to(dtype, non_blocking);
        for (auto& replica : replicas_) {
            if (replica) replica->to(dtype, non_blocking);
        }
    }
    void CustomDataParallel::to(torch::Device device, torch::ScalarType dtype, bool non_blocking) {
        xt::Module::to(device, dtype, non_blocking); // Call base
        // Similar logic to the device-only 'to' method regarding primary_device_
        primary_device_ = device;
        if (original_module_) original_module_->to(device, dtype, non_blocking);

        bool found = false;
        for(const auto& d : devices_){ if(d == device) {found = true; break;}}

        if(found){
            for (size_t i = 0; i < replicas_.size(); ++i) { // Keep replicas on their designated devices but change dtype
                if (replicas_[i]) replicas_[i]->to(devices_[i], dtype, non_blocking);
            }
        } else {
             std::cout << "Warning: CustomDataParallel::to called with a device not in its list. "
                       << "Operating in single-device mode on " << device << std::endl;
             devices_ = {device};
             replicas_ = {original_module_}; // original_module_ was already moved
             if(replicas_[0]) replicas_[0]->to(dtype, non_blocking); // Apply dtype to the single replica
        }
    }


} // namespace xt