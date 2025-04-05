#pragma once

#include "../headers/transforms.h"


namespace xt::data::transforms {
    std::function<torch::Tensor(torch::Tensor input)> create_resize_transform(std::vector<int64_t> size);

    torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size);

    torch::data::transforms::Lambda<torch::data::Example<> > resize(std::vector<int64_t> size);

    torch::data::transforms::Lambda<torch::data::Example<> > normalize(double mean, double stddev);






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







}
