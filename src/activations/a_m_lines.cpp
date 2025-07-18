#include <activations/a_m_lines.h>

namespace xt::activations
{
    torch::Tensor am_lines(const torch::Tensor& x, double negative_slope, double threshold, double high_positive_slope)
    {
        torch::Tensor neg_condition = x.lt(0);
        torch::Tensor mid_condition_upper_bound = x.lt(threshold);

        torch::Tensor neg_values = negative_slope * x;
        torch::Tensor mid_values = x;
        torch::Tensor high_values = threshold + high_positive_slope * (x - threshold);

        torch::Tensor result = torch::where(neg_condition, neg_values, torch::where(
                                                mid_condition_upper_bound, mid_values, high_values));

        return result;
    }


    auto AMLines::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::am_lines(torch::zeros(10));
    }
}
