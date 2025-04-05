#pragma once

#include "../headers/transforms.h"


namespace xt::data::transforms {
    std::function<torch::Tensor(torch::Tensor input)> create_resize_transform(std::vector<int64_t> size);

    torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size);

    torch::data::transforms::Lambda<torch::data::Example<> > resize(std::vector<int64_t> size);

    torch::data::transforms::Lambda<torch::data::Example<> > normalize(double mean, double stddev);










    // struct Rotation {
    // public:
    //     Rotation(float angle);
    //
    //     torch::Tensor operator()(torch::Tensor input);
    //
    // private:
    //     float angle;
    // };



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
