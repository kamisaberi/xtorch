#pragma once
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <string>
#include <vector>
#include <base/module.h>




// namespace xt
// {
//     class Cloneable: public torch::nn::Cloneable<Cloneable> {
//     public:
//         Cloneable();
//         virtual  torch::Tensor forward(torch::Tensor input) const = 0;
//         torch::Tensor operator()(torch::Tensor input) const ;
//         void reset() override;
//
//
//     };
// }



namespace xt {
    // Template class to make modules clonable
    template <typename Derived>
    class Cloneable : public Module {
    public:
        // Constructor
        Cloneable() = default;

        // Override clone method to make the module clonable
        std::shared_ptr<xt::Module> clone() const  {
            // Create a new instance of the derived class
            auto new_module = std::make_shared<Derived>(static_cast<const Derived&>(*this));

            // Copy parameters
            auto src_params = this->named_parameters();
            auto dst_params = new_module->named_parameters();
            for (const auto& param : src_params) {
                auto dst_param = dst_params.find(param.key());
                if (dst_param != nullptr) {
                    dst_param->copy_(param.value().clone());
                }
            }

            // Copy buffers
            auto src_buffers = this->named_buffers();
            auto dst_buffers = new_module->named_buffers();
            for (const auto& buffer : src_buffers) {
                auto dst_buffer = dst_buffers.find(buffer.key());
                if (dst_buffer != nullptr) {
                    dst_buffer->copy_(buffer.value().clone());
                }
            }

            return new_module;
        }

        // Reset method to be overridden by derived classes
        virtual void reset() = 0;
    };
}