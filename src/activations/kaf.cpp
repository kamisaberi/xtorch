#include "include/activations/kaf.h"

namespace xt::activations
{
    torch::Tensor kaf(const torch::Tensor& x,
                                 const torch::Tensor& dictionary_coefs, // Shape [D] or [1, D] for broadcasting
                                 const torch::Tensor& boundary_params // Shape [D-1] or [1, D-1] for broadcasting
    )
    {
        TORCH_CHECK(dictionary_coefs.dim() <= 2, "dictionary_coefs must be 1D or 2D");
        TORCH_CHECK(boundary_params.dim() <= 2, "boundary_params must be 1D or 2D");

        int64_t D = dictionary_coefs.size(-1);
        TORCH_CHECK(D > 0, "Dictionary size D must be greater than 0");
        if (D > 1)
        {
            TORCH_CHECK(boundary_params.size(-1) == D - 1,
                        "boundary_params must have D-1 elements. Got ", boundary_params.size(-1), " for D=", D);
        }

        torch::Tensor x_expanded = x.unsqueeze(-1); // Add a dimension for broadcasting with boundaries

        torch::Tensor result = torch::zeros_like(x);

        if (D == 1)
        {
            result = dictionary_coefs.item<double>() * x; // Single linear function
            return result;
        }

        // Ensure boundary_params are sorted for correct region identification
        // This is crucial. If they are not pre-sorted, this step is needed.
        // For a simple function, we assume they are passed sorted.
        // torch::Tensor sorted_boundaries = torch::sort(boundary_params, -1).values;
        // For this function, let's assume boundary_params are already sorted.
        torch::Tensor sorted_boundaries = boundary_params;


        // Region 0: x < boundary[0]
        torch::Tensor region_0_mask = x_expanded < sorted_boundaries.index({torch::indexing::Slice(), 0});
        result.masked_scatter_(region_0_mask,
                               dictionary_coefs.index({torch::indexing::Slice(), 0}) * x.masked_select(region_0_mask));

        // Intermediate regions: boundary[i-1] <= x < boundary[i]
        for (int64_t i = 1; i < D - 1; ++i)
        {
            torch::Tensor lower_bound = sorted_boundaries.index({torch::indexing::Slice(), i - 1});
            torch::Tensor upper_bound = sorted_boundaries.index({torch::indexing::Slice(), i});
            torch::Tensor region_i_mask = (x_expanded >= lower_bound) & (x_expanded < upper_bound);
            result.masked_scatter_(region_i_mask,
                                   dictionary_coefs.index({torch::indexing::Slice(), i}) * x.masked_select(
                                       region_i_mask));
        }

        // Region D-1: x >= boundary[D-2]
        torch::Tensor region_last_mask = x_expanded >= sorted_boundaries.index({torch::indexing::Slice(), D - 2});
        result.masked_scatter_(region_last_mask,
                               dictionary_coefs.index({torch::indexing::Slice(), D - 1}) * x.masked_select(
                                   region_last_mask));

        return result;
    }

    auto KAF::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::kaf(torch::zeros(10),torch::zeros(10),torch::zeros(10));
    }
}
