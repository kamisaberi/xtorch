#include "../../include/transforms/cutout.h"

namespace xt::data::transforms {

    Cutout::Cutout(int num_holes_, int hole_size_)
        : num_holes(num_holes_), hole_size(hole_size_) {}

    torch::Tensor Cutout::operator()(const torch::Tensor& input_tensor) const {
        auto tensor = input_tensor.clone();
        const int C = tensor.size(0);
        const int H = tensor.size(1);
        const int W = tensor.size(2);

        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> h_dist(0, H - 1);
        std::uniform_int_distribution<> w_dist(0, W - 1);

        for (int n = 0; n < num_holes; ++n) {
            int y = h_dist(gen);
            int x = w_dist(gen);

            int y1 = std::max(0, y - hole_size / 2);
            int y2 = std::min(H, y + hole_size / 2);
            int x1 = std::max(0, x - hole_size / 2);
            int x2 = std::min(W, x + hole_size / 2);

            tensor.slice(1, y1, y2).slice(2, x1, x2).zero_();
        }

        return tensor;
    }




}