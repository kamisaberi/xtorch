#pragma once

#include "../headers/transforms.h"

namespace xt::transforms {

    /**
     * @class Compose
     * @brief A class to compose multiple tensor transformations into a single callable pipeline.
     *
     * The Compose class allows chaining of multiple transformation functions, each operating on a
     * `torch::Tensor`, into a single operation. It is designed to facilitate preprocessing or
     * augmentation of tensor data (e.g., images) by applying a sequence of transforms in the order
     * they are provided. The transformations are stored as a vector of function objects and applied
     * via the call operator.
     */
    class Compose : xt::Module{
    public:
        /**
         * @brief Alias for a transformation function that takes a tensor and returns a tensor.
         *
         * This type alias defines a function signature for transformations that operate on
         * `torch::Tensor` objects, enabling flexible composition of operations.
         */
        using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;

        /**
         * @brief Default constructor, initializing an empty transformation pipeline.
         *
         * Creates a Compose object with no transformations, allowing subsequent addition of transforms
         * if needed.
         */
        Compose();

        /**
         * @brief Constructs a Compose object with a vector of transformation functions.
         * @param transforms A vector of TransformFunc objects specifying the sequence of transformations.
         *
         * Initializes the Compose object with a predefined set of transformations to be applied in order.
         */
        explicit Compose(std::vector<TransformFunc> transforms);

        /**
         * @brief Applies the sequence of transformations to the input tensor.
         * @param input The input tensor to be transformed.
         * @return A tensor resulting from applying all transformations in sequence.
         *
         * This operator applies each transformation in the `transforms` vector to the input tensor,
         * passing the output of one transformation as the input to the next, and returns the final result.
         */
        torch::Tensor operator()(torch::Tensor input) const;

    private:
        /**
         * @brief Vector storing the sequence of transformation functions.
         *
         * This member holds the list of transformations to be applied when the object is called.
         */
        std::vector<TransformFunc> transforms;
    };
}