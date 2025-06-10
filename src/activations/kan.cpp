#include "include/activations/kan.h"

namespace xt::activations
{
    torch::Tensor b_spline_basis(const torch::Tensor& x, const torch::Tensor& grid, int k_order)
    {
        TORCH_CHECK(k_order >= 1, "B-spline order k_order must be >= 1");
        TORCH_CHECK(grid.dim() == 1, "Grid must be 1D");
        TORCH_CHECK(grid.size(0) >= 2 * k_order -1, "Grid size must be at least 2*k_order - 1 for order ", k_order);
        // More stringent check: grid.size(0) must be >= k_order + (k_order-1) + 1 = 2k_order for at least one spline of degree k_order-1
        // Number of basis functions of order k (degree k-1) is grid.size(0) - k_order. This must be > 0.
        TORCH_CHECK(grid.size(0) > k_order, "Grid size must be greater than k_order to form any splines.");


        torch::Tensor x_r = x.contiguous().view({-1, 1});

        int p_degree = k_order - 1;

        // Degree 0 (k_order = 1) B-splines: B_{i,0}(x)
        // B_{i,0}(x) = 1 if grid[i] <= x < grid[i+1] else 0
        auto t_i_deg0 = grid.slice(0, 0, grid.size(0) - 1).unsqueeze(0);
        auto t_i_plus_1_deg0 = grid.slice(0, 1, grid.size(0)).unsqueeze(0);
        torch::Tensor b_current_degree_splines = ((x_r >= t_i_deg0) & (x_r < t_i_plus_1_deg0)).to(x.options());

        // Iteratively compute for higher degrees up to p_degree
        for (int current_deg = 1; current_deg <= p_degree; ++current_deg)
        {
            std::vector<torch::Tensor> next_degree_splines_list;
            // Number of splines of 'current_deg': grid.size(0) - 1 - current_deg
            // b_current_degree_splines has splines of degree (current_deg - 1)
            // It has (grid.size(0) - 1 - (current_deg - 1)) = (grid.size(0) - current_deg) columns

            int num_splines_for_current_deg = grid.size(0) - 1 - current_deg;
            if (num_splines_for_current_deg <= 0 && current_deg > 0)
            {
                // This case should ideally be caught by initial grid size checks if p_degree is too high for the grid.
                // If p_degree is 0, this loop is skipped, b_current_degree_splines is already degree 0.
                TORCH_WARN("Not enough grid points to form splines of degree ", current_deg,
                           ". Returning lower degree splines.");
                break;
            }

            for (int i = 0; i < num_splines_for_current_deg; ++i)
            {
                torch::Tensor term1_num = x_r - grid[i];
                torch::Tensor term1_den_scalar = grid[i + current_deg] - grid[i];

                torch::Tensor term2_num = grid[i + current_deg + 1] - x_r;
                torch::Tensor term2_den_scalar = grid[i + current_deg + 1] - grid[i + 1];

                torch::Tensor b_i_prev_deg = b_current_degree_splines.slice(1, i, i + 1);
                torch::Tensor b_i_plus_1_prev_deg = b_current_degree_splines.slice(1, i + 1, i + 2);

                torch::Tensor term1 = torch::zeros_like(x_r);
                if (term1_den_scalar.item<double>() != 0.0)
                {
                    term1 = (term1_num / term1_den_scalar) * b_i_prev_deg;
                }

                torch::Tensor term2 = torch::zeros_like(x_r);
                if (term2_den_scalar.item<double>() != 0.0)
                {
                    term2 = (term2_num / term2_den_scalar) * b_i_plus_1_prev_deg;
                }
                next_degree_splines_list.push_back(term1 + term2);
            }

            if (next_degree_splines_list.empty())
            {
                if (p_degree == 0) break; // Correct for degree 0 initial case.
                TORCH_WARN("B-spline computation resulted in empty list for degree ", current_deg,
                           ". This should not happen with valid inputs.");
                break;
            }
            b_current_degree_splines = torch::cat(next_degree_splines_list, 1);
        }

        return b_current_degree_splines;
    }


    torch::Tensor kan(
        const torch::Tensor& x,
        const torch::Tensor& spline_weights,
        const torch::Tensor& grid_internal,
        int k_order,
        double base_activation_weight
    )
    {
        TORCH_CHECK(grid_internal.dim() == 1, "Internal grid must be 1D");
        TORCH_CHECK(spline_weights.dim() == 1, "Spline weights must be 1D");

        int G_intervals = grid_internal.size(0) - 1;
        TORCH_CHECK(G_intervals > 0, "Number of grid intervals G_intervals must be > 0");
        TORCH_CHECK(k_order >= 1, "B-spline order k_order must be >= 1");

        int expected_num_spline_coeffs = G_intervals + k_order - 1;
        TORCH_CHECK(spline_weights.size(0) == expected_num_spline_coeffs,
                    "spline_weights size mismatch. Expected ", expected_num_spline_coeffs, " got ",
                    spline_weights.size(0));

        int extended_grid_size = G_intervals + 1 + 2 * (k_order - 1);
        if (k_order == 1) extended_grid_size = G_intervals + 1; // Degree 0: grid is just internal knots

        torch::Tensor extended_grid = torch::empty({extended_grid_size}, grid_internal.options());

        if (k_order == 1)
        {
            // Degree 0
            for (int i = 0; i < G_intervals + 1; ++i)
            {
                extended_grid[i] = grid_internal[i];
            }
        }
        else
        {
            // Degree > 0
            for (int i = 0; i < k_order - 1; ++i)
            {
                extended_grid[i] = grid_internal[0];
                extended_grid[G_intervals + k_order + i] = grid_internal[G_intervals];
            }
            for (int i = 0; i < G_intervals + 1; ++i)
            {
                extended_grid[k_order - 1 + i] = grid_internal[i];
            }
        }

        torch::Tensor b_splines_values = b_spline_basis(x, extended_grid, k_order);

        TORCH_CHECK(b_splines_values.size(1) == expected_num_spline_coeffs,
                    "B-spline basis function output size mismatch. Expected ", expected_num_spline_coeffs,
                    " columns, got ", b_splines_values.size(1),
                    ". x_shape:", x.sizes(), " extended_grid_shape:", extended_grid.sizes(), " k_order:", k_order);

        torch::Tensor spline_component = torch::matmul(b_splines_values, spline_weights);

        torch::Tensor base_component = base_activation_weight * x;
        // torch::Tensor base_component = base_activation_weight * torch::silu(x); // As per paper's base choice

        return spline_component + base_component;
    }


    auto KAN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::kan(torch::zeros(10), torch::zeros(10), torch::zeros(10), 0, 0);
    }
}
