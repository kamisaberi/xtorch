#include <activations/hermite.h>

namespace xt::activations
{
    torch::Tensor hermite(const torch::Tensor& x) {
        torch::Tensor x_abs = torch::abs(x);

        torch::Tensor y_gt_2 = x;

        torch::Tensor y_1_to_2_abs = x_abs * (2.0 - x_abs);
        torch::Tensor y_1_to_2 = torch::copysign(y_1_to_2_abs, x);

        torch::Tensor y_lt_1_abs = (2.0 * x_abs - 3.0) * x_abs * x_abs + 1.0;
        torch::Tensor y_lt_1 = torch::copysign(y_lt_1_abs, x);


        torch::Tensor result = torch::where(
            x_abs > 2.0,
            y_gt_2,
            torch::where(
                x_abs > 1.0,
                y_1_to_2,
                y_lt_1
            )
        );
        return result;
    }
    auto Hermite::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::hermite(torch::zeros(10));
    }
}
