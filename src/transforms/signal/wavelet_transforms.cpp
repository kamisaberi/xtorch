#include "include/transforms/signal/wavelet_transforms.h"
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

/* ... Example Usage ... */

namespace xt::transforms::signal {

    // ... get_wavelet_coeffs and constructor are correct ...

    WaveletTransform::WaveletTransform(const std::string& wavelet, int n_levels, const std::string& padding_mode)
            : n_levels_(n_levels), padding_mode_(padding_mode) {

        // 1. Get the low-pass filter coefficients.
        auto lo_coeffs = get_wavelet_coeffs(wavelet);
        dec_lo_ = torch::tensor(lo_coeffs, torch::kFloat32);

        // 2. Derive the high-pass filter using the Quadrature Mirror Filter (QMF) condition.
        dec_hi_ = torch::flip(dec_lo_, {0});
        auto alternating_sign = torch::pow(-1.0, torch::arange(0, dec_hi_.size(0), torch::kFloat32));
        dec_hi_ *= alternating_sign;

        // 3. Reshape filters for conv1d: (out_channels, in_channels, kernel_size)
        dec_lo_ = dec_lo_.view({1, 1, -1});
        dec_hi_ = dec_hi_.view({1, 1, -1});
    }

    auto WaveletTransform::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("WaveletTransform::forward received an empty list.");
        }
        torch::Tensor waveform = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!waveform.defined() || waveform.dim() != 1) {
            throw std::invalid_argument("Input must be a 1D waveform tensor.");
        }

        auto device = waveform.device();
        dec_lo_ = dec_lo_.to(device);
        dec_hi_ = dec_hi_.to(device);

        long filter_len = dec_lo_.size(-1);
        torch::Tensor current_signal = waveform.clone();
        std::vector<torch::Tensor> coeffs;

        // --- 2. Determine Number of Levels ---
        int max_levels = 0;
        if (filter_len > 1) {
            max_levels = static_cast<int>(std::log2(static_cast<double>(current_signal.size(0)) / (filter_len - 1)));
            max_levels = std::max(0, max_levels);
        }
        int levels = (n_levels_ > 0) ? std::min(n_levels_, max_levels) : max_levels;

        // --- 3. Mallat Algorithm: Iterative Decomposition ---
        for (int i = 0; i < levels; ++i) {
            long signal_len = current_signal.size(0);
            if (signal_len < filter_len) {
                break;
            }

            // --- THE FIX: Use `auto` for the mode variable ---
            // 1. Determine the padding mode enum from the string.
            torch::nn::functional::PadFuncOptions::mode_t mode;
            // auto mode = torch::kZeros; // Correctly deduce the type
            if (padding_mode_ == "reflect") {
                mode = torch::kReflect;
            } else if (padding_mode_ == "replicate") {
                mode = torch::kReplicate;
            } else if (padding_mode_ == "circular") {
                mode = torch::kCircular;
            }

            // 2. Calculate padding amount for DWT.
            long pad_amount = filter_len - 1;
            long pad_left = pad_amount / 2;
            long pad_right = pad_amount - pad_left;

            // 3. Apply padding using torch::nn::functional::pad
            auto input_3d = current_signal.view({1, 1, -1});
            auto padded_input = torch::nn::functional::pad(
                input_3d,
                torch::nn::functional::PadFuncOptions({pad_left, pad_right}).mode(mode)
            );

            // 4. Convolve the PADDED signal with zero padding in the convolution itself, and downsample.
            auto conv_options = torch::nn::functional::Conv1dFuncOptions().stride(2).padding(0);
            auto cA = torch::nn::functional::conv1d(padded_input, dec_lo_, conv_options);
            auto cD = torch::nn::functional::conv1d(padded_input, dec_hi_, conv_options);
            // --- END OF FIX ---

            coeffs.push_back(cD.squeeze());
            current_signal = cA.squeeze();
        }

        // --- 4. Finalize and Order Coefficients ---
        coeffs.push_back(current_signal);
        std::reverse(coeffs.begin(), coeffs.end());

        return coeffs;
    }
} // namespace xt::transforms::signal