#include "include/transforms/target/quantile_transformer.h"

#include <stdexcept>
#include <cmath>       // For std::erf, std::sqrt
#include <algorithm>   // For std::upper_bound

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

void test_transformer(xt::transforms::target::QuantileTransformer& transformer, double value) {
    std::cout << "Input: " << value;
    auto transformed_any = transformer.forward({value});
    auto transformed_value = std::any_cast<double>(transformed_any);
    std::cout << ", Transformed Output: " << transformed_value << std::endl;
}

int main() {
    std::cout.precision(10);

    // 1. --- Setup ---
    // Let's say we "fit" a transformer on our training data and found the
    // following values at the deciles (10 quantiles).
    std::vector<double> learned_deciles = {
        10.0, 25.0, 38.0, 50.0, 65.0, 80.0, 100.0, 125.0, 150.0, 200.0
    };

    // --- Example 1: Transform to a UNIFORM distribution ---
    std::cout << "--- Testing Uniform Output ---" << std::endl;
    xt::transforms::target::QuantileTransformer uniform_transformer(learned_deciles, xt::transforms::target::QuantileOutputType::UNIFORM);
    test_transformer(uniform_transformer, 5.0);   // Below the first quantile -> close to 0
    test_transformer(uniform_transformer, 50.0);  // The 4th quantile value -> should map to ~0.4
    test_transformer(uniform_transformer, 250.0); // Above the last quantile -> close to 1

    // --- Example 2: Transform to a NORMAL distribution ---
    std::cout << "\n--- Testing Normal Output ---" << std::endl;
    xt::transforms::target::QuantileTransformer normal_transformer(learned_deciles, xt::transforms::target::QuantileOutputType::NORMAL);
    test_transformer(normal_transformer, 5.0);   // Should map to a large negative number
    test_transformer(normal_transformer, 50.0);  // ~0.4 quantile -> should map near -0.25
    test_transformer(normal_transformer, 112.5); // The median between two quantiles -> should map near 0
    test_transformer(normal_transformer, 250.0); // Should map to a large positive number

    return 0;
}
*/

namespace xt::transforms::target {

    QuantileTransformer::QuantileTransformer(const std::vector<double>& learned_quantiles, QuantileOutputType output_type)
            : learned_quantiles_(learned_quantiles), output_type_(output_type) {

        if (learned_quantiles_.empty()) {
            throw std::invalid_argument("learned_quantiles vector cannot be empty.");
        }
        // A robust implementation would also check if the vector is sorted.
    }

    auto QuantileTransformer::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("QuantileTransformer::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        double x;

        if (input_any.type() == typeid(double)) {
            x = std::any_cast<double>(input_any);
        } else if (input_any.type() == typeid(float)) {
            x = static_cast<double>(std::any_cast<float>(input_any));
        } else if (input_any.type() == typeid(int)) {
            x = static_cast<double>(std::any_cast<int>(input_any));
        } else if (input_any.type() == typeid(long)) {
            x = static_cast<double>(std::any_cast<long>(input_any));
        } else {
            throw std::invalid_argument("Input to QuantileTransformer must be a scalar numeric type.");
        }

        // 2. --- Find the quantile of the input value ---
        // Find the first element in the learned quantiles that is greater than x.
        auto it = std::upper_bound(learned_quantiles_.begin(), learned_quantiles_.end(), x);

        // The rank is its distance from the beginning.
        double rank = std::distance(learned_quantiles_.begin(), it);

        // To handle interpolation between quantiles, we can add a fractional part.
        if (it != learned_quantiles_.begin() && it != learned_quantiles_.end()) {
            double upper_bound = *it;
            double lower_bound = *(it - 1);
            if (upper_bound > lower_bound) {
                rank -= (upper_bound - x) / (upper_bound - lower_bound);
            }
        }

        // Normalize the rank to a percentile in the range [0, 1]
        double percentile = rank / learned_quantiles_.size();

        // Clamp to avoid issues with the inverse normal CDF
        percentile = std::max(1e-7, std::min(1.0 - 1e-7, percentile));

        // 3. --- Map to the target distribution ---
        if (output_type_ == QuantileOutputType::UNIFORM) {
            return percentile;
        } else { // NORMAL
            return inverse_normal_cdf(percentile);
        }
    }

    // Peter John Acklam's approximation for the inverse normal CDF.
    // This is a common and reasonably accurate approximation used when a full
    // stats library isn't available.
    double QuantileTransformer::inverse_normal_cdf(double p) {
        if (p <= 0.0 || p >= 1.0) {
            // Out of domain - return +/- infinity or a large number
            return (p > 0.5) ? 1e9 : -1e9;
        }

        constexpr double a1 = -39.69683028665376;
        constexpr double a2 = 220.9460984245205;
        constexpr double a3 = -275.9285104469687;
        constexpr double a4 = 138.3577518672690;
        constexpr double a5 = -30.66479806614716;
        constexpr double a6 = 2.506628277459239;

        constexpr double b1 = -54.47609879822406;
        constexpr double b2 = 161.5858368580409;
        constexpr double b3 = -155.6989798598866;
        constexpr double b4 = 66.80131188771972;
        constexpr double b5 = -13.28068155288572;

        constexpr double c1 = -7.784894002430293E-03;
        constexpr double c2 = -0.3223964580411365;
        constexpr double c3 = -2.400758277161838;
        constexpr double c4 = -2.549732539343734;
        constexpr double c5 = 4.374664141464968;
        constexpr double c6 = 2.938163982698783;

        constexpr double d1 = 7.784695709041462E-03;
        constexpr double d2 = 0.3224671290700398;
        constexpr double d3 = 2.445134137142996;
        constexpr double d4 = 3.754408661907416;

        constexpr double p_low = 0.02425;
        constexpr double p_high = 1.0 - p_low;

        double q, r;
        if (p < p_low) {
            q = std::sqrt(-2.0 * std::log(p));
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        } else if (p <= p_high) {
            q = p - 0.5;
            r = q * q;
            return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
                   (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
        } else {
            q = std::sqrt(-2.0 * std::log(1.0 - p));
            return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }
    }

} // namespace xt::transforms::target