#include "include/transforms/appliers/repeat.h"

namespace xt::transforms {
    Repeat::Repeat() = default;

    Repeat::Repeat(std::unique_ptr <xt::Module> transform, int n_times) : xt::Module(), transform(std::move(transform)),
                                                                          n_times_(n_times) {
    }

    auto Repeat::forward(std::initializer_list <torch::Tensor> tensors) -> std::any {
        std::vector <torch::Tensor> tensor_vec(tensors);
        torch::Tensor input = tensor_vec[0];
        for (int i = 1; i < this->n_times_; i++) {
            // input = transform(std::move(input));
        }
        return input;


    }

}
