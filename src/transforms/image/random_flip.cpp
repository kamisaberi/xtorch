#include "../../../include/transforms/image/random_flip.h"

namespace xt::transforms::image {

    RandomFlip::RandomFlip(double h_prob, double v_prob)
        : horizontal_prob(h_prob), vertical_prob(v_prob) {
    }

    torch::Tensor RandomFlip::operator()(const torch::Tensor &input_tensor) {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::bernoulli_distribution flip_h(horizontal_prob);
        std::bernoulli_distribution flip_v(vertical_prob);

        // Convert CHW -> HWC
        auto img_tensor = input_tensor.detach().cpu().clone().permute({1, 2, 0});
        img_tensor = img_tensor.mul(255).clamp(0, 255).to(torch::kU8);

        cv::Mat img(img_tensor.size(0), img_tensor.size(1), CV_8UC3);
        std::memcpy(img.data, img_tensor.data_ptr(), sizeof(uint8_t) * img_tensor.numel());

        if (flip_h(gen)) {
            cv::flip(img, img, 1); // Horizontal
        }
        if (flip_v(gen)) {
            cv::flip(img, img, 0); // Vertical
        }

        torch::Tensor output = torch::from_blob(
            img.data, {img.rows, img.cols, 3}, torch::kUInt8).clone();

        output = output.permute({2, 0, 1}).to(torch::kFloat32).div(255); // HWC -> CHW
        return output;
    }





}