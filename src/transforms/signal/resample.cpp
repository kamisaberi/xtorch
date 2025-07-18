#include <transforms/signal/resample.h>

#include <samplerate.h>


/*
// Example Usage (goes in a main.cpp or test file)
#include "include/utils/audio/io.h"
#include <iostream>

int main() {
    // 1. Load a high sample rate audio file (e.g., 44100 Hz).
    auto [waveform_44k, sr_44k] = xt::utils::audio::load("audio_file_44100hz.wav");
    std::cout << "Loaded waveform with shape: " << waveform_44k.sizes()
              << " at " << sr_44k << " Hz" << std::endl;

    // 2. Create a Resample transform to downsample to 16000 Hz.
    // Use the default highest quality setting.
    xt::transforms::signal::Resample resampler(sr_44k, 16000);

    // 3. Apply the transform.
    torch::Tensor waveform_16k = std::any_cast<torch::Tensor>(
        resampler.forward({waveform_44k})
    );

    // 4. Verify the output.
    std::cout << "Resampled waveform shape: " << waveform_16k.sizes() << std::endl;
    // The length will be proportional to the change in sample rate.
    // e.g., length_16k ~= length_44k * (16000 / 44100)

    // 5. Save the result.
    xt::utils::audio::save("resampled_to_16k.wav", waveform_16k, 16000);

    return 0;
}
*/

namespace xt::transforms::signal {

    Resample::Resample(int orig_freq, int new_freq, int quality, double p)
            : orig_freq_(orig_freq), new_freq_(new_freq), quality_(quality), p_(p) {

        if (orig_freq_ <= 0 || new_freq_ <= 0) {
            throw std::invalid_argument("Original and new sample rates must be positive.");
        }
        if (quality_ < 0 || quality_ > 4) {
            throw std::invalid_argument("Quality must be between 0 and 4.");
        }
        is_identity_ = (orig_freq_ == new_freq_);

        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto Resample::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation and Probability ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Resample::forward received an empty list.");
        }
        torch::Tensor waveform = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!waveform.defined()) {
            throw std::invalid_argument("Input tensor passed to Resample is not defined.");
        }

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (is_identity_ || prob_dist(random_engine_) > p_) {
            return waveform; // Skip transform
        }

        // --- 2. Prepare for Resampling ---
        // Ensure tensor is on CPU and float for libsamplerate
        waveform = waveform.to(torch::kCPU, torch::kFloat32).contiguous();

        auto waveform_dims = waveform.dim();
        if (waveform_dims < 1 || waveform_dims > 2) {
            throw std::invalid_argument("Resample only supports 1D or 2D tensors.");
        }

        // --- 3. Perform Resampling (iterating over channels if needed) ---
        int num_channels = (waveform_dims == 1) ? 1 : waveform.size(0);
        std::vector<torch::Tensor> resampled_channels;
        resampled_channels.reserve(num_channels);

        for (int c = 0; c < num_channels; ++c) {
            auto channel_waveform = (waveform_dims == 1) ? waveform : waveform.select(0, c);
            const float* input_ptr = channel_waveform.data_ptr<float>();
            long input_frames = channel_waveform.size(0);

            double src_ratio = static_cast<double>(new_freq_) / static_cast<double>(orig_freq_);
            long output_frames_estimate = static_cast<long>(input_frames * src_ratio) + 10;
            std::vector<float> output_vector(output_frames_estimate);

            SRC_DATA src_data;
            src_data.data_in = input_ptr;
            src_data.input_frames = input_frames;
            src_data.data_out = output_vector.data();
            src_data.output_frames = output_frames_estimate;
            src_data.src_ratio = src_ratio;

            // `src_simple` is the easiest API for this. It handles state internally.
            // We use 1 channel here because we are iterating.
            int error = src_simple(&src_data, quality_, 1);
            if (error) {
                throw std::runtime_error(
                        "libsamplerate error: " + std::string(src_strerror(error))
                );
            }

            // The actual number of output frames might be different from the estimate.
            output_vector.resize(src_data.output_frames_gen);
            resampled_channels.push_back(torch::tensor(output_vector));
        }

        // --- 4. Finalize Output ---
        if (num_channels == 1) {
            return resampled_channels[0].to(waveform.device());
        } else {
            return torch::stack(resampled_channels, 0).to(waveform.device());
        }
    }

} // namespace xt::transforms::signal