
#include <transforms/general/convert_dtype.h>

namespace xt::transforms::general {
    ConvertDType::ConvertDType() = default;

    ConvertDType::ConvertDType(torch::ScalarType target_dtype) : xt::Module(), target_dtype(target_dtype) {
    }


    auto ConvertDType::forward(std::initializer_list <std::any> tensors) -> std::any {

        // 1. Convert the initializer_list to a std::vector to safely access elements.
        std::vector<std::any> any_vec(tensors);

        // 2. Add a robustness check for an empty input list.
        if (any_vec.empty()) {
            throw std::invalid_argument("ConvertDType::forward received an empty list of tensors.");
        }

        // 3. Safely cast the first element from std::any to torch::Tensor.
        torch::Tensor input = std::any_cast<torch::Tensor>(any_vec[0]);

        // 4. Ensure the tensor is valid before proceeding.
        if (!input.defined()) {
            throw std::invalid_argument("Input tensor passed to ConvertDType is not defined.");
        }

        // 5. (Optional but good practice) If the tensor already has the target dtype,
        // return it without performing a no-op conversion.
        if (input.dtype() == target_dtype) {
            return input;
        }

        // 6. Perform the conversion using the .to() method and return the new tensor.
        // The result is automatically wrapped in std::any on return.
        return input.to(target_dtype);
    }

}
