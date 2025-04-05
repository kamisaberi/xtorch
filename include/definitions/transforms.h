#pragma once

#include "../headers/transforms.h"


namespace xt::data::transforms {
    std::function<torch::Tensor(torch::Tensor input)> create_resize_transform(std::vector<int64_t> size);

    torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size);

    torch::data::transforms::Lambda<torch::data::Example<> > resize(std::vector<int64_t> size);

    torch::data::transforms::Lambda<torch::data::Example<> > normalize(double mean, double stddev);


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


    struct GaussianBlurOpenCV {
    public:
        GaussianBlurOpenCV(int ksize, double sigma_val);

        torch::Tensor operator()(const torch::Tensor &input_tensor);

    private:
        cv::Size kernel_size;
        double sigma;
    };


    struct RandomGaussianBlur {
    private:
        std::vector<int> kernel_sizes; // List of odd kernel sizes to choose from
        double sigma_min;
        double sigma_max;

    public:
        RandomGaussianBlur(std::vector<int> sizes = {3, 5}, double sigma_min = 0.1, double sigma_max = 2.0);

        torch::Tensor operator()(const torch::Tensor &input_tensor);
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

    struct Rotation {
    public:
        Rotation(double angle_deg);

        torch::Tensor operator()(const torch::Tensor &input_tensor);

    private :
        double angle; // Rotation angle in degrees
    };


    struct RandomCrop2 {
    private:
        int crop_height;
        int crop_width;

    public:
        RandomCrop2(int height, int width);

        torch::Tensor operator()(const torch::Tensor &input_tensor);
    };


    struct RandomFlip {
    private:
        double horizontal_prob;
        double vertical_prob;

    public:
        RandomFlip(double h_prob = 0.5, double v_prob = 0.0);

        torch::Tensor operator()(const torch::Tensor &input_tensor);
    };


    struct ToTensor {
    public:
        torch::Tensor operator()(const cv::Mat& image) const;
    };


    struct Normalize {
    public:
        Normalize(std::vector<float> mean, std::vector<float> std);
        torch::Tensor operator()(const torch::Tensor& tensor) const;
    private:
        std::vector<float> mean;
        std::vector<float> std;

    };


    struct ToGray {
        torch::Tensor operator()(const torch::Tensor& color_tensor) const;
    };





}
