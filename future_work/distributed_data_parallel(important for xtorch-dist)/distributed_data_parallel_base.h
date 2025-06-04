#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory> // For std::shared_ptr
#include <any>    // For std::any
#include <string>

#include "include/base/base.h" // Assuming this is where your xt::Module is defined

namespace xt {
namespace nn {
namespace parallel {

    /**
     * @class DistributedDataParallelBase
     * @brief Abstract base class for Distributed Data Parallel modules.
     *
     * This class defines the common interface for modules that distribute training
     * across multiple processes/devices. Concrete implementations will wrap a given
     * xt::Module and handle the distributed aspects like model replication,
     * gradient synchronization, and data sharding (though data sharding is
     * typically handled by a distributed DataLoader/Sampler).
     *
     * This module itself should inherit from xt::Module (or indirectly if xt::Module
     * inherits from torch::nn::Module) to be compatible with the rest of the
     * xt::
     * ecosystem.
     */
    class DistributedDataParallelBase : public xt::Module { // Inherits from your base xt::Module
    public:
        // --- Constructor & Destructor ---

        /**
         * @brief Constructor.
         * @param module_to_wrap The underlying xt::Module to be parallelized.
         *                       This DDP module will typically manage this wrapped module.
         * @param process_group A handle or representation of the distributed process group.
         *                      The exact type might vary (e.g., void* for opaque handle,
         *                      or a specific process group class). For simplicity, we omit
         *                      it here but a real implementation would need it.
         *                      Alternatively, rank/world_size are passed for initialization.
         * @param rank The rank of the current process in the distributed group.
         * @param world_size The total number of processes in the distributed group.
         * @param device The primary torch::Device this rank's module replica will reside on.
         */
        DistributedDataParallelBase(std::shared_ptr<xt::Module> module_to_wrap,
                                    int rank,
                                    int world_size,
                                    torch::Device device)
            : underlying_module_ptr_(module_to_wrap),
              rank_(rank),
              world_size_(world_size),
              device_(device) {
            if (!underlying_module_ptr_) {
                throw std::invalid_argument("Module to wrap in DistributedDataParallelBase cannot be null.");
            }
            // The base DDP class might move the underlying module to its device.
            // Concrete implementations will handle replication.
            underlying_module_ptr_->to(device_);
        }

        virtual ~DistributedDataParallelBase() override = default;

        // --- Core DDP Methods (Pure Virtual - Must be implemented by derived classes) ---

        /**
         * @brief Performs the forward pass.
         * The input data in `inputs` is typically assumed to be the shard for the current rank.
         * The DDP module handles scattering inputs if it receives a global batch (less common for DDP)
         * or assumes the DataLoader provides sharded input.
         * This method will execute the forward pass on the local replica and potentially
         * participate in collective operations if the forward pass itself is distributed
         * (e.g., for model parallelism combined with data parallelism, though not the primary DDP mode).
         * For standard DDP, this is mostly a local forward pass.
         * @param inputs An initializer list of std::any, typically containing the input tensor(s) for this rank.
         * @return std::any containing the output tensor(s) from this rank's forward pass.
         */
        virtual std::any forward(std::initializer_list<std::any> inputs) override = 0;

        /**
         * @brief Synchronizes gradients across all processes in the group.
         * This method is typically called *after* `loss.backward()` has been executed
         * on the local loss for the current rank's micro-batch.
         * It ensures that gradients are averaged (or summed) across all replicas.
         */
        virtual void synchronize_gradients() = 0;

        // --- Standard xt::Module/torch::nn::Module Interface Overrides ---
        // These methods should operate on or reflect the state of the *underlying wrapped module*
        // but ensure consistency if DDP manages multiple replicas.

        /**
         * @brief Returns the parameters of the underlying model.
         * In DDP, all replicas should have synchronized parameters, so returning
         * parameters from any (e.g., the local or primary) replica is valid.
         * @param recurse If true, recursively include parameters of submodules.
         * @return A vector of Tensors representing the model parameters.
         */
        std::vector<torch::Tensor> parameters(bool recurse = true) const override {
            return underlying_module_ptr_->parameters(recurse);
        }

        /**
         * @brief Returns the named parameters of the underlying model.
         * @param recurse If true, recursively include named parameters of submodules.
         * @return An OrderedDict of parameter names to Tensors.
         */
        torch::OrderedDict<std::string, torch::Tensor> named_parameters(bool recurse = true) const override {
            return underlying_module_ptr_->named_parameters(recurse);
        }

        /**
         * @brief Moves and/or casts the parameters and buffers.
         * For DDP, this is complex. It might mean re-initializing the DDP setup if the
         * set of devices changes, or moving all replicas. For simplicity, this might
         * primarily affect the local replica and require explicit re-synchronization.
         * The `device_` member should be updated.
         * @param device The target device.
         * @param non_blocking If true, and the source data is in pinned memory and
         *                     destination is CUDA, the copy is performed asynchronously.
         */
        void to(torch::Device device, bool non_blocking = false) override {
            xt::Module::to(device, non_blocking); // Call base if it has general logic
            device_ = device;
            underlying_module_ptr_->to(device_, non_blocking);
            // Concrete DDP implementation might need to rebroadcast parameters here if world_size > 1
            // or re-initialize internal replica structures.
            if (world_size_ > 1) {
                // This is a placeholder for more complex DDP re-sync logic on `to`
                std::cout << "Warning: DDPBase::to(Device) called. Parameter re-synchronization across ranks might be needed." << std::endl;
                // A full implementation might re-broadcast parameters from rank 0.
                // For example: request_parameter_broadcast();
            }
        }

        void to(torch::ScalarType dtype, bool non_blocking = false) override {
            xt::Module::to(dtype, non_blocking);
            underlying_module_ptr_->to(dtype, non_blocking);
            if (world_size_ > 1) {
                 std::cout << "Warning: DDPBase::to(dtype) called. Parameter re-synchronization across ranks might be needed." << std::endl;
            }
        }

        void to(torch::Device device, torch::ScalarType dtype, bool non_blocking = false) override {
            xt::Module::to(device, dtype, non_blocking);
            device_ = device;
            underlying_module_ptr_->to(device_, dtype, non_blocking);
            if (world_size_ > 1) {
                 std::cout << "Warning: DDPBase::to(Device, dtype) called. Parameter re-synchronization across ranks might be needed." << std::endl;
            }
        }

        /**
         * @brief Sets the module in training mode.
         * This should be propagated to the underlying wrapped module(s).
         * @param on True for training mode, false for evaluation mode.
         */
        void train(bool on = true) override {
            xt::Module::train(on); // Call base if it has general logic
            underlying_module_ptr_->train(on);
            // Concrete DDP might have other state to set (e.g., related to gradient sync hooks)
        }

        /**
         * @brief Sets the module in evaluation mode.
         */
        void eval() override {
            train(false);
        }

        /**
         * @brief Zeros out the gradients of all parameters of the underlying model.
         * @param set_to_none If true, gradients will be set to undefined tensors.
         */
        void zero_grad(bool set_to_none = true) override {
             xt::Module::zero_grad(set_to_none); // Call base if it has general logic
            underlying_module_ptr_->zero_grad(set_to_none);
        }

        // --- Accessors and Properties ---

        /**
         * @brief Gets a pointer to the underlying (original) module being wrapped.
         * Useful for saving the model (typically only rank 0 saves the underlying module's state_dict).
         * @return A shared_ptr to the underlying xt::Module.
         */
        std::shared_ptr<xt::Module> module() const {
            return underlying_module_ptr_;
        }

        /**
         * @brief Gets the rank of the current process.
         * @return The rank.
         */
        int rank() const { return rank_; }

        /**
         * @brief Gets the total number of processes in the distributed group.
         * @return The world size.
         */
        int world_size() const { return world_size_; }

        /**
         * @brief Gets the primary device this DDP rank is operating on.
         * @return The torch::Device.
         */
        torch::Device device() const { return device_; }


        // --- Potentially other useful methods for DDP management ---
        // virtual void rebuild_buckets() {} // For DDP implementations that use gradient bucketing
        // virtual bool find_unused_parameters() const { return false; } // If supporting this DDP option
        // virtual void set_find_unused_parameters(bool find) {}

    protected:
        std::shared_ptr<xt::Module> underlying_module_ptr_; // The actual model logic
        int rank_;
        int world_size_;
        torch::Device device_; // Device this rank's primary replica is on

        // Placeholder for a method that concrete classes might use to trigger a parameter broadcast
        // This would typically be called after 'to' if parameters might be out of sync.
        // virtual void request_parameter_broadcast() {
        //     if (rank_ == 0) { /* initiate broadcast */ }
        //     else { /* participate in broadcast */ }
        // }
    };

} // namespace parallel
} // namespace nn
} // namespace xt