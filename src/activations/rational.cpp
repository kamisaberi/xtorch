#include <activations/rational.h>

namespace xt::activations
{
    torch::Tensor rational(
        const torch::Tensor& x,
        const torch::Tensor& P_coeffs, // Numerator coefficients [p_m, ..., p_1, p_0]
        const torch::Tensor& Q_coeffs, // Denominator coefficients [q_n, ..., q_1] (q_0 is fixed to 1)
        double epsilon // For denominator stability
    )
    {
        TORCH_CHECK(P_coeffs.dim() == 1, "P_coeffs (numerator) must be a 1D tensor.");
        TORCH_CHECK(Q_coeffs.dim() <= 1, "Q_coeffs (denominator) must be a 0D (scalar q0=1) or 1D tensor.");
        // Allow Q_coeffs to be empty if Q(x)=1
        TORCH_CHECK(P_coeffs.size(0) > 0, "Numerator must have at least one coefficient (p_0).");

        int64_t m_degree = P_coeffs.size(0) - 1;

        torch::Tensor numerator = torch::zeros_like(x);
        for (int64_t i = 0; i <= m_degree; ++i)
        {
            numerator += P_coeffs[m_degree - i] * torch::pow(x, static_cast<double>(i));
        }

        torch::Tensor denominator = torch::ones_like(x); // q_0 * x^0 where q_0 = 1
        if (Q_coeffs.size(0) > 0)
        {
            int64_t n_degree_q_terms = Q_coeffs.size(0); // Number of q_j terms for j >= 1
            for (int64_t i = 0; i < n_degree_q_terms; ++i)
            {
                // Q_coeffs are [q_n, ..., q_1]
                denominator += Q_coeffs[n_degree_q_terms - 1 - i] * torch::pow(x, static_cast<double>(i + 1));
            }
        }
        // The paper's formulation for stability often ensures Q(x) > 0.
        // E.g. Q(x) = 1 + sum |q_j x^j| or other positive definite forms.
        // For this simple version, we rely on epsilon, but a robust Q might be needed in practice.
        // For general Q_coeffs, the denominator could become zero or negative.
        // To match common practice for PAU/Rational Activations for stability:
        // Q(x) = 1 + sum_{j=1 to n} |q_j * x^j|
        // Recalculate denominator using this more robust form if Q_coeffs are provided:
        if (Q_coeffs.size(0) > 0)
        {
            denominator = torch::ones_like(x); // q_0 = 1
            int64_t n_degree_q_terms = Q_coeffs.size(0);
            for (int64_t i = 0; i < n_degree_q_terms; ++i)
            {
                denominator += torch::abs(
                    Q_coeffs[n_degree_q_terms - 1 - i] * torch::pow(x, static_cast<double>(i + 1)));
            }
        }


        return numerator / (denominator + epsilon);
    }

    auto Rational::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::rational(torch::zeros(10), torch::zeros(10), torch::zeros(10));
    }
}
