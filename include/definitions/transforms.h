#pragma once

#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include <torch/data/transforms/base.h>
#include <functional>

namespace xt::data::transforms {
    torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size);

    torch::Tensor pad_tensor(const torch::Tensor &tensor, const int size);

    torch::Tensor grayscale_image(const torch::Tensor &tensor);

    torch::Tensor grayscale_to_rgb(const torch::Tensor &tensor);


    torch::data::transforms::Lambda<torch::data::Example<> > resize(std::vector<int64_t> size);

    torch::data::transforms::Lambda<torch::data::Example<> > pad(int size);

    torch::data::transforms::Lambda<torch::data::Example<> > grayscale();

    torch::data::transforms::Lambda<torch::data::Example<> > normalize(double mean, double stddev);

    torch::data::transforms::Lambda<torch::data::Example<> > grayscaleToRGB();


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
    class Compose {
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
        Compose(std::vector<TransformFunc> transforms);

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


    /**
     * @struct Resize
     * @brief A functor to resize a tensor image to a specified size.
     *
     * This struct provides a callable object that resizes a `torch::Tensor` representing an image
     * to a target size specified as a vector of 64-bit integers. It uses the call operator to
     * perform the resizing operation, making it suitable for use in functional pipelines or
     * transformations.
     */
    struct Resize {
    public:
        /**
         * @brief Constructs a Resize object with the target size.
         * @param size A vector of 64-bit integers specifying the target dimensions (e.g., {height, width}).
         */
        Resize(std::vector<int64_t> size);

        /**
         * @brief Resizes the input tensor image to the target size.
         * @param img The input tensor image to be resized.
         * @return A new tensor with the resized dimensions.
         */
        torch::Tensor operator()(torch::Tensor img);

    private:
        std::vector<int64_t> size; ///< The target size for resizing (e.g., {height, width}).
    };

    std::function<torch::Tensor(torch::Tensor input)> create_resize_transform(std::vector<int64_t> size);
}
