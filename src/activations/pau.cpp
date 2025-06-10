#include "include/activations/pau.h"

namespace xt::activations
{
    torch::Tensor pau(
        const torch::Tensor& x,
        const torch::Tensor& P_coeffs, // Numerator coefficients [p_m, p_{m-1}, ..., p_1, p_0]
        const torch::Tensor& Q_coeffs, // Denominator coefficients [q_n, q_{n-1}, ..., q_1, 1.0] (q_0 is fixed to 1)
        double epsilon
    )
    {
        TORCH_CHECK(P_coeffs.dim() == 1, "P_coeffs (numerator) must be a 1D tensor.");
        TORCH_CHECK(Q_coeffs.dim() == 1, "Q_coeffs (denominator) must be a 1D tensor.");
        TORCH_CHECK(P_coeffs.size(0) > 0, "Numerator must have at least one coefficient (p_0).");
        TORCH_CHECK(Q_coeffs.size(0) > 0, "Denominator must have at least one coefficient (q_0=1).");

        int64_t m_degree = P_coeffs.size(0) - 1; // Degree of numerator P(x)
        int64_t n_degree = Q_coeffs.size(0) - 1; // Degree of denominator Q(x) (excluding fixed q_0=1)

        torch::Tensor numerator = torch::zeros_like(x);
        for (int64_t i = 0; i <= m_degree; ++i)
        {
            numerator += P_coeffs[m_degree - i] * torch::pow(x, static_cast<double>(i));
        }

        torch::Tensor denominator = torch::ones_like(x); // q_0 * x^0 where q_0 = 1
        for (int64_t i = 0; i <= n_degree - 1; ++i)
        {
            // Q_coeffs are [q_n, ..., q_1]
            denominator += Q_coeffs[n_degree - 1 - i] * torch::pow(x, static_cast<double>(i + 1));
        }

        // Add epsilon to denominator for numerical stability
        // The paper mentions ensuring Q(x) > 0, often by Q(x) = 1 + |sum q_i x^i|.
        // For a simple function, we'll just add epsilon.
        // A more robust Q might be Q(x) = 1 + sum_{i=1 to n} |q_i| |x|^i or similar.
        // Or, as paper suggests, Q(x) = 1 + sum_{j=1..n} |q_j x^j| using abs values.
        // For this simple version, let's stick to the direct polynomial sum + epsilon.
        // However, if Q_coeffs can be negative, Q(x) can be zero or negative.
        // The paper's official implementation uses Q(x) = 1 + sum_j |a_j * x^j|, which is always >= 1.
        // Let's modify to reflect that for better stability for general coefficients Q.

        // Recalculate denominator based on Q(x) = 1 + sum_{j=1 to n} |q_j * x^j| approach
        denominator = torch::ones_like(x); // q_0 = 1
        for (int64_t i = 0; i <= n_degree - 1; ++i)
        {
            // Q_coeffs are [q_n, ..., q_1]
            denominator += torch::abs(Q_coeffs[n_degree - 1 - i] * torch::pow(x, static_cast<double>(i + 1)));
        }


        return numerator / (denominator + epsilon);
    }

    auto PAU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::pau(torch::zeros(10), torch::zeros(10), torch::zeros(10));
    }
}
