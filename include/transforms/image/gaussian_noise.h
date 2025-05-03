
#include "../headers/transforms.h"

namespace xt::data::transforms {


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


}