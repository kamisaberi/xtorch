#include "include/utils/rabdom.h"
#include <cmath>
#include <random> // For standard C++ random number generation

namespace xt::utils::random {

    // A thread-local random number generator is good practice for multi-threaded applications.
    thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    double sample_gamma(double alpha) {
        if (alpha <= 0.0) {
            throw std::invalid_argument("Alpha must be positive for Gamma distribution.");
        }

        // Use different algorithms for alpha < 1 and alpha >= 1
        if (alpha < 1.0) {
            // Ahrens and Dieter's method for 0 < alpha < 1
            double u = uniform_dist(generator);
            return sample_gamma(1.0 + alpha) * std::pow(u, 1.0 / alpha);
        }

        // Marsaglia and Tsang's method for alpha >= 1
        double d, c, x, v, u;
        d = alpha - 1.0 / 3.0;
        c = 1.0 / std::sqrt(9.0 * d);
        while (true) {
            do {
                // Sample from a standard normal distribution
                // Using Box-Muller transform for simplicity
                double u1 = uniform_dist(generator);
                double u2 = uniform_dist(generator);
                x = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
                v = 1.0 + c * x;
            } while (v <= 0.0);

            v = v * v * v;
            u = uniform_dist(generator);
            if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
                return (d * v);
            }
            if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) {
                return (d * v);
            }
        }
    }

    double sample_beta(double alpha, double beta) {
        if (alpha <= 0.0 || beta <= 0.0) {
            throw std::invalid_argument("Alpha and beta must be positive for Beta distribution.");
        }

        double x = sample_gamma(alpha);
        double y = sample_gamma(beta);
        return x / (x + y);
    }

} // namespace xt::utils::random