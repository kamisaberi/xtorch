#include "tensor_converter.h"

template <typename T>
TensorConverter<T>::TensorConverter() = default;

template <typename T>
torch::Tensor TensorConverter<T>::to_tensor(const std::vector<T>& data,
                                            const std::vector<int64_t>& shape,
                                            torch::ScalarType dtype,
                                            torch::Device device) {
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

template <typename T>
torch::Tensor TensorConverter<T>::to_tensor(const T* data,
                                            size_t size,
                                            const std::vector<int64_t>& shape,
                                            torch::ScalarType dtype,
                                            torch::Device device) {
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

template <typename T>
torch::Tensor TensorConverter<T>::convert_dtype(const torch::Tensor& tensor,
                                                torch::ScalarType target_dtype) {
    if (!tensor.defined()) {
        throw std::invalid_argument("Input tensor is not defined");
    }
    if (tensor.dtype() == target_dtype) {
        return tensor;
    }
    return tensor.to(target_dtype);
}

// Explicit template instantiations for common types
template class TensorConverter<float>;
template class TensorConverter<double>;
template class TensorConverter<int32_t>;
template class TensorConverter<int64_t>;