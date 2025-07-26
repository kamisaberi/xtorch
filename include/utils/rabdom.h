#pragma once

#include <torch/torch.h>

namespace xt::utils::random {

    /**
     * @brief Generates a random number from a Gamma(alpha, 1.0) distribution.
     *
     * Uses Marsaglia and Tsang's method for alpha >= 1.
     * For 0 < alpha < 1, it uses the method from Ahrens and Dieter.
     * This provides a robust way to sample from the Gamma distribution.
     *
     * @param alpha The shape parameter of the Gamma distribution. Must be positive.
     * @return A single double sampled from the distribution.
     */
    double sample_gamma(double alpha);


    /**
     * @brief Generates a random number from a Beta(alpha, beta) distribution.
     *
     * This is implemented by sampling two Gamma variables and combining them.
     *
     * @param alpha The first shape parameter of the Beta distribution. Must be positive.
     * @param beta The second shape parameter of the Beta distribution. Must be positive.
     * @return A single double sampled from the distribution, in the range [0, 1].
     */
    double sample_beta(double alpha, double beta);

} // namespace xt::utils::random