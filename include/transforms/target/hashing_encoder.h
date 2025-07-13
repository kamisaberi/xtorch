#pragma once

#include "../common.h"


#include <torch/torch.h> // Assumes LibTorch is installed
#include <string>
#include <functional> // For std::hash

namespace xt::transforms::target {

    /**
     * @class HashingEncoder
     * @brief A target transformation that maps a categorical label to a fixed-size
     *        vector using the "hashing trick".
     *
     * This technique avoids the need for a pre-built vocabulary by using a hash
     * function to map the input category directly to an index in a vector of
     * fixed dimension. It's highly scalable for features with many categories.
     */
    class HashingEncoder : public xt::Module {
    public:
        /**
         * @brief Constructs the HashingEncoder.
         *
         * @param num_dimensions The desired number of output dimensions (the size
         *                       of the hash space). A larger number reduces the
         *                       chance of hash collisions.
         */
        explicit HashingEncoder(int num_dimensions);

        /**
         * @brief Executes the hashing and encoding operation.
         * @param tensors An initializer list expected to contain a single label.
         *                The label can be a `std::string`, `int`, `long`, etc.
         *                It will be converted to a string before hashing.
         * @return An std::any containing the hashed feature as a 1D `torch::Tensor`
         *         of shape (num_dimensions,). This tensor is all zeros except
         *         for a 1 at the calculated hash index.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int num_dimensions_;
        std::hash<std::string> hasher_;
    };

} // namespace xt::transforms::target