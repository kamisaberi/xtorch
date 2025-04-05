#include "../../include/transforms/compose.h"

namespace xt::data::transforms {

/**
 * @brief Alias for a transformation function that takes a tensor and returns a tensor.
 *
 * This type alias defines a function signature for transformations that operate on
 * `torch::Tensor` objects, enabling flexible composition of operations within the Compose class.
 */
using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;

/**
 * @brief Default constructor, initializing an empty transformation pipeline.
 *
 * Creates a Compose object with no transformations, allowing subsequent addition of transforms
 * if needed. The internal vector of transformations is default-initialized to empty.
 */
Compose::Compose() {
}

/**
 * @brief Constructs a Compose object with a vector of transformation functions.
 * @param transforms A vector of TransformFunc objects specifying the sequence of transformations.
 *
 * Initializes the Compose object by storing the provided vector of transformation functions,
 * which will be applied in sequence when the object is called.
 */
Compose::Compose(std::vector<TransformFunc> transforms)
    : transforms(transforms) {
}

/**
 * @brief Applies the sequence of transformations to the input tensor.
 * @param input The input tensor to be transformed.
 * @return A tensor resulting from applying all transformations in sequence.
 *
 * This function iterates over the stored transformations, applying each one to the input tensor
 * in the order they were provided. Each transformation’s output becomes the input to the next,
 * with the final result returned. The input tensor is moved into each transformation to optimize
 * performance by avoiding unnecessary copies where possible.
 */
torch::Tensor Compose::operator()(torch::Tensor input) const {
    for (const auto &transform: this->transforms) {
        input = transform(std::move(input));
    }
    return input;
}

}