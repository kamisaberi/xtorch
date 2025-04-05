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

    torch::data::transforms::Lambda<torch::data::Example<> > resize(std::vector<int64_t> size) {
        return torch::data::transforms::Lambda<torch::data::Example<> >(
            [size](torch::data::Example<> example) {
                example.data = resize_tensor(example.data, size);
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
     * @brief Converts a grayscale tensor to an RGB tensor.
     * @param tensor The input grayscale tensor, expected in format [N, H, W] or [N, 1, H, W].
     * @return A new tensor in RGB format [N, 3, H, W], with the grayscale values replicated across channels.
     *
     * This function transforms a grayscale tensor into an RGB tensor by ensuring the input has a channel
     * dimension and then replicating that channel three times to form RGB channels. If the input tensor
     * is 3D ([N, H, W]), it adds a channel dimension to make it [N, 1, H, W]. If it’s already 4D ([N, 1, H, W]),
     * it uses it as is. The `repeat` operation then duplicates the single channel into three, producing
     * an output tensor of shape [N, 3, H, W], where N is the batch size, H is height, and W is width.
     * This is useful for converting batched grayscale images to RGB format in LibTorch workflows.
     */
    torch::Tensor GrayscaleToRGB::operator()(const torch::Tensor &tensor) {
        torch::Tensor gray = tensor.dim() == 3 ? tensor.unsqueeze(1) : tensor; // Ensure [N, 1, H, W]
        return gray.repeat({1, 3, 1, 1}); // [N, 1, H, W] -> [N, 3, H, W]
    }


    /**
     * @brief Constructs a Resize object with the target size.
     * @param size A vector of 64-bit integers specifying the target dimensions (e.g., {height, width}).
     *
     * Initializes the Resize object by storing the provided size vector, which will be used
     * to resize input tensors in subsequent calls to the operator() function.
     */
    Resize::Resize(std::vector<int64_t> size) : size(size) {
    }

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


    /**
     * @brief Constructs a Pad object with the specified padding sizes.
     * @param padding A vector of 64-bit integers defining the padding amounts, in pairs (e.g., {left, right, top, bottom}).
     *
     * Initializes the Pad object by storing the provided padding vector, which will be used to pad
     * input tensors in subsequent calls to the operator() function. The vector must contain an even
     * number of elements, where each pair specifies the left and right padding for a dimension.
     * No validation is performed in this implementation; invalid padding sizes may result in runtime
     * errors when applied.
     */
    Pad::Pad(std::vector<int64_t> padding) : padding(padding) {
    }

    /**
     * @brief Applies padding to the input tensor using the stored padding configuration.
     * @param input The input tensor to be padded, typically in format [N, C, H, W] or [H, W].
     * @return A new tensor with padded dimensions according to the stored padding configuration.
     *
     * This function pads the input tensor using LibTorch’s torch::nn::functional::pad utility with
     * the padding sizes specified during construction. The padding is applied with constant mode
     * (defaulting to zeros) to the last dimensions of the tensor, as determined by the number of
     * pairs in the padding vector. For example, for a 4D tensor [N, C, H, W] with padding {p_left,
     * p_right, p_top, p_bottom}, it pads width (W) and height (H), resulting in [N, C, H + p_top +
     * p_bottom, W + p_left + p_right]. The number of padding values must be even and compatible
     * with the tensor’s dimensions, or a runtime error will occur.
     */
    torch::Tensor Pad::operator()(torch::Tensor input) {
        return torch::nn::functional::pad(input, padding);
    }


    CenterCrop::CenterCrop(std::vector<int64_t> size) : size(size) {
        if (size.size() != 2) {
            throw std::invalid_argument("CenterCrop size must have exactly 2 elements (height, width).");
        }
    }

    torch::Tensor CenterCrop::operator()(torch::Tensor input) {
        int64_t input_dims = input.dim();
        if (input_dims < 2) {
            throw std::runtime_error("Input tensor must have at least 2 dimensions for cropping.");
        }

        // Get input height and width (last two dimensions)
        int64_t input_h = input.size(input_dims - 2);
        int64_t input_w = input.size(input_dims - 1);
        int64_t target_h = size[0];
        int64_t target_w = size[1];

        // Validate input size is large enough
        if (input_h < target_h || input_w < target_w) {
            throw std::runtime_error("Input dimensions must be >= target size for cropping.");
        }

        // Calculate crop start and end indices
        int64_t h_start = (input_h - target_h) / 2;
        int64_t h_end = h_start + target_h;
        int64_t w_start = (input_w - target_w) / 2;
        int64_t w_end = w_start + target_w;

        // Crop height (dim -2) and width (dim -1)
        return input.slice(input_dims - 2, h_start, h_end)
                .slice(input_dims - 1, w_start, w_end);
    }


    Grayscale::Grayscale() {
    }

    torch::Tensor Grayscale::operator()(torch::Tensor input) {
        int64_t input_dims = input.dim();
        if (input_dims < 3) {
            throw std::runtime_error("Input tensor must have at least 3 dimensions (e.g., [C, H, W]).");
        }

        // Get channel dimension (assumed as dim 0 or dim 1 for batched)
        int64_t channel_dim = (input_dims == 3) ? 0 : 1;
        int64_t channels = input.size(channel_dim);
        if (channels != 3) {
            throw std::runtime_error("Input tensor must have exactly 3 channels (RGB).");
        }

        // Define grayscale weights (ITU-R 601-2 luma transform)
        auto weights = torch::tensor({0.2989, 0.5870, 0.1140},
                                     torch::TensorOptions().dtype(input.dtype()).device(input.device()));

        // Compute weighted sum across channels
        torch::Tensor gray = (input * weights.view({channels, 1, 1})).sum(channel_dim, true);
        return gray; // Output shape: e.g., [1, H, W] or [N, 1, H, W]
    }






struct GaussianBlur {
public:
    // Constructor: Initialize kernel size and sigma, validate inputs
    GaussianBlur(std::vector<int64_t> kernel_size, float sigma)
        : kernel_size(kernel_size), sigma(sigma) {
        if (kernel_size.size() != 2) {
            throw std::invalid_argument("Kernel size must have exactly 2 elements (height, width).");
        }
        if (kernel_size[0] % 2 == 0 || kernel_size[1] % 2 == 0) {
            throw std::invalid_argument("Kernel dimensions must be odd.");
        }
        if (sigma <= 0) {
            throw std::invalid_argument("Sigma must be positive.");
        }
    }

    // Operator: Apply Gaussian blur to the input tensor
    torch::Tensor operator()(torch::Tensor input) {
        int64_t input_dims = input.dim();
        if (input_dims < 3 || input_dims > 4) {
            throw std::runtime_error("Input tensor must be 3D ([C, H, W]) or 4D ([N, C, H, W]).");
        }

        // Determine channels and ensure input format
        int64_t channel_dim = (input_dims == 3) ? 0 : 1;
        int64_t channels = input.size(channel_dim);
        if (channels < 1) {
            throw std::runtime_error("Input must have at least one channel.");
        }

        // Generate 2D Gaussian kernel
        int64_t k_h = kernel_size[0];
        int64_t k_w = kernel_size[1];
        torch::Tensor kernel = generate_gaussian_kernel(k_h, k_w, sigma, input.device());

        // Reshape kernel for conv2d: [out_channels, in_channels/groups, k_h, k_w]
        kernel = kernel.unsqueeze(0).unsqueeze(0); // [1, 1, k_h, k_w]
        kernel = kernel.repeat({channels, 1, 1, 1}); // [C, 1, k_h, k_w]

        // Add batch dimension if 3D
        bool is_3d = (input_dims == 3);
        if (is_3d) {
            input = input.unsqueeze(0); // [1, C, H, W]
        }

        // Apply convolution with "same" padding
        torch::Tensor output = torch::conv2d(input, kernel,
                                            /*bias=*/torch::Tensor(),
                                            /*stride=*/1,
                                            /*padding=*/{(k_h - 1) / 2, (k_w - 1) / 2},
                                            /*dilation=*/1,
                                            /*groups=*/channels);

        // Remove batch dimension if added
        if (is_3d) {
            output = output.squeeze(0); // [C, H, W]
        }

        return output;
    }

private:
    std::vector<int64_t> kernel_size; // Kernel size {height, width}
    float sigma;                      // Gaussian sigma

    // Helper to generate 2D Gaussian kernel
    torch::Tensor generate_gaussian_kernel(int64_t k_h, int64_t k_w, float sigma, torch::Device device) {
        torch::Tensor x = torch::arange(-(k_w / 2), k_w / 2 + 1, torch::dtype(torch::kFloat32).device(device));
        torch::Tensor y = torch::arange(-(k_h / 2), k_h / 2 + 1, torch::dtype(torch::kFloat32).device(device));
        auto [x_grid, y_grid] = torch::meshgrid({x, y}, "ij");
        torch::Tensor kernel = torch::exp(-(x_grid.pow(2) + y_grid.pow(2)) / (2 * sigma * sigma));
        kernel = kernel / kernel.sum(); // Normalize
        return kernel;
    }
};








}
