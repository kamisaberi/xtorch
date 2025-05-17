#pragma once
#include <torch/torch.h>
#include <vector>
#include <stdexcept>

template <typename T>
class TensorConverter {
public:
    // Constructor
    TensorConverter() = default;

    // Convert std::vector to torch::Tensor
    torch::Tensor to_tensor(const std::vector<T>& data,
                            const std::vector<int64_t>& shape,
                            torch::ScalarType dtype = torch::typeMetaToScalarType(torch::CppTypeToTensorType<T>::value),
                            torch::Device device = torch::kCPU) {
        if (data.empty()) {
            throw std::invalid_argument("Input vector is empty");
        }

        // Verify that the shape matches the data size
        int64_t total_size = 1;
        for (auto dim : shape) {
            total_size *= dim;
        }
        if (total_size != static_cast<int64_t>(data.size())) {
            throw std::invalid_argument("Shape does not match data size");
        }

        // Create tensor from data
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        return torch::from_blob(const_cast<T*>(data.data()), shape, options).clone();
    }

    // Convert raw array to torch::Tensor
    torch::Tensor to_tensor(const T* data,
                            size_t size,
                            const std::vector<int64_t>& shape,
                            torch::ScalarType dtype = torch::typeMetaToScalarType(torch::CppTypeToTensorType<T>::value),
                            torch::Device device = torch::kCPU) {
        if (!data || size == 0) {
            throw std::invalid_argument("Invalid input array or zero size");
        }

        // Verify that the shape matches the data size
        int64_t total_size = 1;
        for (auto dim : shape) {
            total_size *= dim;
        }
        if (total_size != static_cast<int64_t>(size)) {
            throw std::invalid_argument("Shape does not match data size");
        }

        // Create tensor from data
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        return torch::from_blob(const_cast<T*>(data), shape, options).clone();
    }

    // Convert tensor to specified dtype
    torch::Tensor convert_dtype(const torch::Tensor& tensor,
                                torch::ScalarType target_dtype) {
        if (!tensor.defined()) {
            throw std::invalid_argument("Input tensor is not defined");
        }
        if (tensor.dtype() == target_dtype) {
            return tensor;
        }
        return tensor.to(target_dtype);
    }

private:
    // Helper to map C++ types to default torch scalar types
    template <typename U>
    struct CppTypeToTensorType {
        static constexpr torch::ScalarType value = torch::kFloat;
    };

    // Specializations for common types
    template <> struct CppTypeToTensorType<float> {
        static constexpr torch::ScalarType value = torch::kFloat;
    };
    template <> struct CppTypeToTensorType<double> {
        static constexpr torch::ScalarType value = torch::kDouble;
    };
    template <> struct CppTypeToTensorType<int32_t> {
        static constexpr torch::ScalarType value = torch::kInt;
    };
    template <> struct CppTypeToTensorType<int64_t> {
        static constexpr torch::ScalarType value = torch::kLong;
    };
};