#include "include/transforms/video/uniform_temporal_subsample.h"

#include <stdexcept>
#include <cmath> // For std::round

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

int main() {
    // 1. --- Create Dummy Video Clip Data ---
    // Create a simple 16-frame clip. The values in each frame will be equal
    // to the frame's index, making the selection easy to verify.
    int total_frames = 16;
    torch::Tensor clip = torch::arange(total_frames, torch::kFloat32).view({total_frames, 1, 1, 1}).expand({total_frames, 1, 2, 2});

    std::cout << "Original clip has " << total_frames << " frames, with indices 0-15." << std::endl;

    // 2. --- Setup the Transform ---
    // We want to subsample 4 frames uniformly from the 16 available frames.
    int num_to_sample = 4;
    xt::transforms::video::UniformTemporalSubsample subsampler(num_to_sample);
    std::cout << "Subsampling to " << num_to_sample << " frames." << std::endl;

    // 3. --- Run the Transform ---
    auto subsampled_any = subsampler.forward({clip});

    // 4. --- Verify the Output ---
    try {
        auto subsampled_clip = std::any_cast<torch::Tensor>(subsampled_any);
        std::cout << "\nOutput clip shape: " << subsampled_clip.sizes() << std::endl;

        // Let's check the values of the frames we got.
        // For T=16, N=4, the step is (16-1)/(4-1) = 15/3 = 5.
        // The indices should be round(0*5)=0, round(1*5)=5, round(2*5)=10, round(3*5)=15.
        std::cout << "Values of the sampled frames (should be 0, 5, 10, 15):" << std::endl;
        std::cout << "[" << subsampled_clip[0].sum().item<float>() / 4.0 // Divide by C*H*W to get original value
                  << ", " << subsampled_clip[1].sum().item<float>() / 4.0
                  << ", " << subsampled_clip[2].sum().item<float>() / 4.0
                  << ", " << subsampled_clip[3].sum().item<float>() / 4.0 << "]" << std::endl;

        if (subsampled_clip.size(0) == num_to_sample) {
            std::cout << "\nVerification successful!" << std::endl;
        }

    } catch (const std::bad_any_cast& e) {
        std::cerr << "Failed to cast the result to torch::Tensor." << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::video {

    UniformTemporalSubsample::UniformTemporalSubsample(int num_samples)
        : num_samples_(num_samples) {
        if (num_samples_ <= 0) {
            throw std::invalid_argument("Number of samples must be positive.");
        }
    }

    auto UniformTemporalSubsample::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("UniformTemporalSubsample::forward received an empty list.");
        }

        torch::Tensor input_clip;
        try {
            input_clip = std::any_cast<torch::Tensor>(any_vec[0]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Input to UniformTemporalSubsample must be a torch::Tensor.");
        }

        if (input_clip.dim() != 4) {
            throw std::invalid_argument("Input clip must be a 4D tensor (T, C, H, W).");
        }

        const long total_frames = input_clip.size(0);
        if (total_frames < num_samples_) {
            throw std::invalid_argument(
                "The number of frames in the clip (" + std::to_string(total_frames) +
                ") is less than the number of frames to sample (" + std::to_string(num_samples_) + ")."
            );
        }

        // If the clip already has the desired number of frames, no work is needed.
        if (total_frames == num_samples_) {
            return input_clip;
        }

        // 2. --- Calculate Indices to Keep ---
        std::vector<long> indices_to_keep;
        indices_to_keep.reserve(num_samples_);

        if (num_samples_ == 1) {
            // If only one frame is needed, take the one from the middle.
            indices_to_keep.push_back(total_frames / 2);
        } else {
            // Calculate the temporal stride between the frames to select.
            // Using a double for precision before rounding.
            const double step = static_cast<double>(total_frames - 1) / (num_samples_ - 1);
            for (int i = 0; i < num_samples_; ++i) {
                long index = std::round(i * step);
                indices_to_keep.push_back(index);
            }
        }

        // 3. --- Perform Indexing ---
        // Convert the vector of indices to a 1D tensor of longs.
        torch::Tensor index_tensor = torch::tensor(indices_to_keep, torch::kLong);

        // Use torch::index_select to efficiently gather the frames along the time dimension (dim=0).
        torch::Tensor output_clip = torch::index_select(input_clip, 0, index_tensor);

        return output_clip;
    }

} // namespace xt::transforms::video