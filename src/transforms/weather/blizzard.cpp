#include <transforms/weather/blizzard.h>


// --- Example Main (for testing) ---
// You can uncomment this to create a standalone executable for testing the transform.
/*
#include <iostream>

int main() {
    // 1. Create dummy data representing a 4x4 grid.
    // Wind Speed in km/h
    torch::Tensor wind_speed = torch::tensor({
        {25, 60, 65, 40},
        {30, 55, 70, 75},
        {58, 20, 30, 60},
        {70, 80, 50, 25}
    }, torch::kFloat32);

    // Snowfall Rate in cm/hr
    torch::Tensor snowfall_rate = torch::tensor({
        {0.2, 0.6, 0.4, 0.8},
        {0.1, 1.0, 1.2, 0.3},
        {0.7, 0.1, 0.9, 0.6},
        {0.8, 0.9, 0.2, 0.1}
    }, torch::kFloat32);

    std::cout << "--- Detecting Blizzard with Default Thresholds (Wind > 56 km/h, Snow > 0.5 cm/hr) ---" << std::endl;
    std::cout << "Input Wind Speed (km/h):\n" << wind_speed << std::endl;
    std::cout << "Input Snowfall Rate (cm/hr):\n" << snowfall_rate << std::endl;

    // 2. Create and apply the transform.
    xt::transforms::weather::Blizzard blizzard_detector;
    torch::Tensor blizzard_map = std::any_cast<torch::Tensor>(blizzard_detector.forward({wind_speed, snowfall_rate}));

    // 3. Print the result.
    std::cout << "\nBlizzard Condition Map (1 = Blizzard, 0 = No Blizzard):\n" << blizzard_map << std::endl;

    // --- Expected Output ---
    // Blizzard Condition Map (1 = Blizzard, 0 = No Blizzard):
    // 0  1  0  0
    // 0  0  1  0
    // 1  0  0  1
    // 1  1  0  0
    // [ CPUFloatType{4,4} ]

    return 0;
}
*/

namespace xt::transforms::weather {

    Blizzard::Blizzard() : wind_speed_threshold_(56.0f), snowfall_rate_threshold_(0.5f) {}

    Blizzard::Blizzard(float wind_speed_threshold, float snowfall_rate_threshold)
            : wind_speed_threshold_(wind_speed_threshold), snowfall_rate_threshold_(snowfall_rate_threshold) {
        if (wind_speed_threshold_ < 0 || snowfall_rate_threshold_ < 0) {
            throw std::invalid_argument("Blizzard thresholds cannot be negative.");
        }
    }

    auto Blizzard::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() < 2) {
            throw std::invalid_argument("Blizzard::forward expects at least two tensors (wind speed and snowfall rate).");
        }

        torch::Tensor wind_speed = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor snowfall_rate = std::any_cast<torch::Tensor>(any_vec[1]);

        if (!wind_speed.defined() || !snowfall_rate.defined()) {
            throw std::invalid_argument("Input tensors passed to Blizzard are not defined.");
        }

        if (!wind_speed.sizes().equals(snowfall_rate.sizes())) {
            throw std::invalid_argument("Input tensors for Blizzard must have the same shape.");
        }

        if (wind_speed.dtype() != snowfall_rate.dtype()) {
            // Or, alternatively, cast one to match the other. For now, we enforce they are the same.
            throw std::invalid_argument("Input tensors for Blizzard must have the same data type.");
        }


        // 2. --- Apply Blizzard Logic ---
        // Use PyTorch's element-wise operators to find where conditions are met.
        // The results of these comparisons are boolean tensors (torch::kBool).
        torch::Tensor high_wind = torch::gt(wind_speed, wind_speed_threshold_);
        torch::Tensor heavy_snow = torch::gt(snowfall_rate, snowfall_rate_threshold_);

        // Perform a logical AND to find where both conditions are true.
        torch::Tensor blizzard_conditions = torch::logical_and(high_wind, heavy_snow);

        // 3. --- Convert back to original data type ---
        // Converts the boolean tensor (true/false) to a numeric tensor (1.0/0.0),
        // matching the type of the input tensors (e.g., kFloat32).
        return blizzard_conditions.to(wind_speed.dtype());
    }

} // namespace xt::transforms::weather