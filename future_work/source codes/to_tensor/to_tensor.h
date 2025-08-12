#pragma once
#include <torch/torch.h>
#include <vector>
#include <stdexcept>

template <typename T>
class TensorConverter {
public:
    // Constructor
    TensorConverter();

    // Convert std::vector to torch::Tensor
    torch::Tensor to_tensor(const std::vector<T>& data,
                            const std::vector<int64_t>& shape,
                            torch::ScalarType dtype = torch::typeMetaToScalarType(torch::CppTypeToTensorType<T>::value),
                            torch::Device device = torch::kCPU);

    // Convert raw array to torch::Tensor
    torch::Tensor to_tensor(const T* data,
                            size_t size,
                            const std::vector<int64_t>& shape,
                            torch::ScalarType dtype = torch::typeMetaToScalarType(torch::CppTypeToTensorType<T>::value),
                            torch::Device device = torch::kCPU);

    // Convert tensor to specified dtype
    torch::Tensor convert_dtype(const torch::Tensor& tensor,
                                torch::ScalarType target_dtype);

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