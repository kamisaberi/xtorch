#pragma once

#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include <torch/data/transforms/base.h>
#include <functional>
#include <stdexcept>
#include <cmath>

namespace xt::data::transforms {
    std::function<torch::Tensor(torch::Tensor input)> create_resize_transform(std::vector<int64_t> size);

    torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size);

    torch::data::transforms::Lambda<torch::data::Example<> > resize(std::vector<int64_t> size);

    torch::data::transforms::Lambda<torch::data::Example<> > normalize(double mean, double stddev);

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
     * @struct GrayscaleToRGB
     * @brief A functor to convert a grayscale tensor to an RGB tensor.
     *
     * This struct provides a callable object that transforms a grayscale tensor, typically with a single
     * channel (e.g., [H, W] or [1, H, W]), into an RGB tensor with three channels (e.g., [3, H, W]).
     * The conversion is performed by replicating the grayscale channel across the RGB dimensions,
     * suitable for preprocessing grayscale images in machine learning workflows using LibTorch.
     */
    struct GrayscaleToRGB {
    public:
        /**
         * @brief Converts a grayscale tensor to an RGB tensor.
         * @param tensor The input grayscale tensor, expected in format [H, W] or [1, H, W].
         * @return A new tensor in RGB format [3, H, W], with the grayscale values replicated across channels.
         *
         * This operator takes a grayscale tensor and produces an RGB tensor by duplicating the single
         * channel’s values into three identical channels (red, green, blue). The input tensor must have
         * a single channel, either as a 2D tensor [H, W] or a 3D tensor with one channel [1, H, W].
         */
        torch::Tensor operator()(const torch::Tensor &tensor);
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


    /**
     * @struct Pad
     * @brief A functor to pad a tensor with a specified padding configuration.
     *
     * This struct provides a callable object that applies padding to a `torch::Tensor` based on a
     * vector of padding sizes. It is designed to extend the dimensions of a tensor (e.g., images
     * in machine learning workflows) by adding values (typically zeros) around its boundaries,
     * using the padding amounts specified during construction. The padding is applied using
     * LibTorch's functional padding utilities.
     */
    struct Pad {
    public:
        /**
         * @brief Constructs a Pad object with the specified padding sizes.
         * @param padding A vector of 64-bit integers defining the padding amounts, in pairs (e.g., {left, right, top, bottom}).
         *
         * Initializes the Pad object with a vector specifying the padding to be applied to the tensor’s
         * dimensions. The vector must contain an even number of elements, where each pair corresponds
         * to the left and right padding for a dimension, applied to the tensor’s last dimensions in
         * reverse order (e.g., width, then height for a 2D tensor).
         */
        Pad(std::vector<int64_t> padding);

        /**
         * @brief Applies padding to the input tensor.
         * @param input The input tensor to be padded, typically in format [N, C, H, W] or [H, W].
         * @return A new tensor with padded dimensions according to the stored padding configuration.
         *
         * This operator pads the input tensor using the padding sizes provided at construction,
         * typically with zeros using constant mode padding. For a 4D tensor [N, C, H, W] and padding
         * {p_left, p_right, p_top, p_bottom}, it pads width (W) and height (H), resulting in
         * [N, C, H + p_top + p_bottom, W + p_left + p_right]. The padding is applied to the last
         * dimensions corresponding to the number of pairs in the padding vector.
         */
        torch::Tensor operator()(torch::Tensor input);

    private:
        /**
         * @brief Vector storing the padding sizes.
         *
         * This member holds the padding configuration, where each pair of values specifies the
         * left and right padding for a dimension of the tensor.
         */
        std::vector<int64_t> padding;
    };


    struct CenterCrop {
    public:
        CenterCrop(std::vector<int64_t> size);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::vector<int64_t> size;
    };


    struct Grayscale {
    public:
        Grayscale();

        torch::Tensor operator()(torch::Tensor input);
    };


    struct GaussianBlur {
    public:
        GaussianBlur(std::vector<int64_t> kernel_size, float sigma);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::vector<int64_t> kernel_size;
        float sigma;

        torch::Tensor generate_gaussian_kernel(int64_t k_h, int64_t k_w, float sigma, torch::Device device);
    };


    struct GaussianNoise {
    public:
        GaussianNoise(float mean, float std);

        torch::Tensor operator()(torch::Tensor input);

    private:
        float mean;
        float std;
    };


    struct HorizontalFlip {
    public:
        HorizontalFlip();

        torch::Tensor operator()(torch::Tensor input);
    };


    struct VerticalFlip {
    public:
        VerticalFlip();

        torch::Tensor operator()(torch::Tensor input);
    };


    struct RandomCrop {
    public:
        RandomCrop(std::vector<int64_t> size);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::vector<int64_t> size;
    };

    struct Lambda {
    public:
        Lambda(std::function<torch::Tensor(torch::Tensor)> transform);

        torch::Tensor operator()(torch::Tensor input);

    private:
        std::function<torch::Tensor(torch::Tensor)> transform;
    };


    // struct Rotation {
    // public:
    //     Rotation(float angle);
    //
    //     torch::Tensor operator()(torch::Tensor input);
    //
    // private:
    //     float angle;
    // };


}
