#include "include/activations/maxout.h"

namespace xt::activations
{
    torch::Tensor maxout(const torch::Tensor& x, int64_t num_pieces, int64_t dim)
    {
        TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension.");
        TORCH_CHECK(dim < x.dim() && dim >= -x.dim(), "Dimension out of range.");
        if (dim < 0)
        {
            dim += x.dim();
        }
        TORCH_CHECK(x.size(dim) % num_pieces == 0,
                    "Dimension ", dim, " size (", x.size(dim),
                    ") must be divisible by num_pieces (", num_pieces, ").");

        std::vector<int64_t> original_shape = x.sizes().vec();
        int64_t target_dim_size = original_shape[dim] / num_pieces;

        std::vector<int64_t> reshape_dims = original_shape;
        reshape_dims.insert(reshape_dims.begin() + dim + 1, num_pieces);
        reshape_dims[dim] = target_dim_size;

        torch::Tensor reshaped_x = x.reshape(reshape_dims);

        torch::Tensor max_values = std::get<0>(torch::max(reshaped_x, dim + 1));

        return max_values;
    }


    auto Maxout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::maxout(torch::zeros(10), 0);
    }
}
