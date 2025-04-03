#include "../../include/definitions/transforms.h"

namespace xt::data::transforms {


    std::function<torch::Tensor(torch::Tensor input)> create_resize_transform(std::vector<int64_t> size) {
        auto resize_fn = [size](torch::Tensor img) -> torch::Tensor {
            img = img.unsqueeze(0); // Add batch dimension
            img = torch::nn::functional::interpolate(
                img,
                torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>({size[0], size[1]}))
                .mode(torch::kBilinear)
                .align_corners(false)
            );
            return img.squeeze(0); // Remove batch dimension
        };
        return resize_fn;
    }





    torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size) {
        return torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions().size(size).mode(torch::kBilinear).align_corners(false)
        ).squeeze(0);
    }

    torch::Tensor pad_tensor(const torch::Tensor &tensor, int size) {
        std::vector<int64_t> ssize = {size, size};
        return torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions().size(ssize).mode(torch::kBilinear).align_corners(false)
        ).squeeze(0);
    }

    torch::Tensor grayscale_image(const torch::Tensor &tensor) {
        return torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear).align_corners(false)
        ).squeeze(0);
    }

    torch::Tensor grayscale_to_rgb(const torch::Tensor &tensor) {
        torch::Tensor gray = tensor.dim() == 3 ? tensor.unsqueeze(1) : tensor; // Ensure [N, 1, H, W]
        return gray.repeat({1, 3, 1, 1}); // [N, 1, H, W] -> [N, 3, H, W]
    }

    torch::data::transforms::Lambda<torch::data::Example<> > resize(std::vector<int64_t> size) {
        return torch::data::transforms::Lambda<torch::data::Example<> >(
            [size](torch::data::Example<> example) {
                example.data = resize_tensor(example.data, size);
                return example;
            }
        );
    }

    torch::data::transforms::Lambda<torch::data::Example<> > pad(int size) {
        return torch::data::transforms::Lambda<torch::data::Example<> >(
            [size](torch::data::Example<> example) {
                example.data = pad_tensor(example.data, size);
                return example;
            }
        );
    }

    torch::data::transforms::Lambda<torch::data::Example<> > grayscale() {
        return torch::data::transforms::Lambda<torch::data::Example<> >(
            [](torch::data::Example<> example) {
                example.data = grayscale_image(example.data);
                return example;
            }
        );
    }

    torch::data::transforms::Lambda<torch::data::Example<> > normalize(double mean, double stddev) {
        return torch::data::transforms::Lambda<torch::data::Example<> >(
            [mean, stddev](torch::data::Example<> example) {
                example.data = example.data.to(torch::kFloat32).div(255);
                return example;
            }
        );
    }


    torch::data::transforms::Lambda<torch::data::Example<> > grayscaleToRGB() {
        return torch::data::transforms::Lambda<torch::data::Example<> >(
            [](torch::data::Example<> example) {
                example.data = grayscale_image(example.data);
                return example;
            }
        );
    }






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




    /**
     * @brief Constructs a Resize object with the target size.
     * @param size A vector of 64-bit integers specifying the target dimensions (e.g., {height, width}).
     *
     * Initializes the Resize object by storing the provided size vector, which will be used
     * to resize input tensors in subsequent calls to the operator() function.
     */
    Resize::Resize(std::vector<int64_t> size) : size(size) {}

    /**
     * @brief Resizes the input tensor image to the target size using bilinear interpolation.
     * @param img The input tensor image to be resized, typically in format [C, H, W] (channels, height, width).
     * @return A new tensor with the resized dimensions, in format [C, H', W'] where H' and W' match the target size.
     *
     * This function applies bilinear interpolation to resize the input image tensor to the dimensions
     * specified in the constructor. It adds a batch dimension before interpolation (making the tensor
     * [1, C, H, W]), resizes it using torch::nn::functional::interpolate, and removes the batch dimension
     * afterward to return a tensor in the original format [C, H', W']. The interpolation is performed
     * with bilinear mode and align_corners set to false for smooth and standard resizing behavior.
     */
    torch::Tensor Resize::operator()(torch::Tensor img) {
        img = img.unsqueeze(0); // Add batch dimension
        img = torch::nn::functional::interpolate(
            img,
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>({size[0], size[1]}))
            .mode(torch::kBilinear)
            .align_corners(false)
        );
        return img.squeeze(0); // Remove batch dimension
    }



}
