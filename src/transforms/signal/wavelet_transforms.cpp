#include "include/transforms/signal/wavelet_transforms.h"


/*
// Example Usage (goes in a main.cpp or test file)
#include "include/utils/audio/io.h"
#include <iostream>

int main() {
    // 1. Load an audio file.
    auto [waveform, sample_rate] = xt::utils::audio::load("some_audio_file.wav");

    // 2. Create the WaveletTransform.
    // Let's use the Daubechies 4 wavelet with 5 levels of decomposition.
    xt::transforms::signal::WaveletTransform wavelet_transform("db4", 5);

    // 3. Apply the transform.
    // The result is a vector of tensors.
    auto result_any = wavelet_transform.forward({waveform});
    auto coeffs = std::any_cast<std::vector<torch::Tensor>>(result_any);

    // 4. Inspect the output.
    // The coefficients are ordered [cA5, cD5, cD4, cD3, cD2, cD1].
    std::cout << "Wavelet decomposition resulted in " << coeffs.size() << " coefficient tensors." << std::endl;
    for (size_t i = 0; i < coeffs.size(); ++i) {
        std::string name = (i == 0) ? "cA" + std::to_string(coeffs.size() - 1)
                                    : "cD" + std::to_string(coeffs.size() - i);
        std::cout << "  - " << name << " shape: " << coeffs[i].sizes() << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::signal {

    // A small database of common wavelet filter coefficients (low-pass decomposition).
    // Source: PyWavelets library / standard wavelet texts.
    std::vector<float> WaveletTransform::get_wavelet_coeffs(const std::string& name) {
        static const std::map<std::string, std::vector<float>> wavelet_db = {
                {"haar", {0.7071067811865476, 0.7071067811865476}},
                {"db2", {0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.12940952255126037}},
                {"db4", {0.2303778133088964, 0.7148465705529154, 0.6308807679298587, -0.027983769416859854,
                                 -0.1870348117190931, 0.030841381835560763, 0.0328830116668852, -0.01059740178506903}}
                // More wavelets (coif, sym, etc.) can be added here.
        };

        auto it = wavelet_db.find(name);
        if (it == wavelet_db.end()) {
            throw std::invalid_argument("Wavelet '" + name + "' is not supported.");
        }
        return it->second;
    }

    WaveletTransform::WaveletTransform(const std::string& wavelet, int n_levels, const std::string& padding_mode)
            : n_levels_(n_levels), padding_mode_(padding_mode) {

        // 1. Get the low-pass filter coefficients.
        auto lo_coeffs = get_wavelet_coeffs(wavelet);
        dec_lo_ = torch::tensor(lo_coeffs, torch::kFloat32);

        // 2. Derive the high-pass filter using the Quadrature Mirror Filter (QMF) condition.
        // g[n] = (-1)^n * h[L - 1 - n]
        dec_hi_ = torch::flip(dec_lo_, {0});
        auto alternating_sign = torch::pow(-1.0, torch::arange(0, dec_hi_.size(0)));
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
            max_levels = static_cast<int>(std::floor(std::log2(current_signal.size(0) / (filter_len - 1.0))));
        }
        int levels = (n_levels_ > 0) ? std::min(n_levels_, max_levels) : max_levels;

        // --- 3. Mallat Algorithm: Iterative Decomposition ---
        for (int i = 0; i < levels; ++i) {
            long signal_len = current_signal.size(0);
            if (signal_len < filter_len) {
                break; // Cannot convolve if signal is shorter than filter
            }

            // The functional version of conv1d handles padding modes like "reflect".
            // Input must be (N, C, L).
            auto input_3d = current_signal.view({1, 1, -1});

            // Convolve and downsample
            auto cA = torch::nn::functional::conv1d(input_3d, dec_lo_,
                                                    torch::nn::functional::Conv1dFuncOptions().stride(2).padding(padding_mode_));

            auto cD = torch::nn::functional::conv1d(input_3d, dec_hi_,
                                                    torch::nn::functional::Conv1dFuncOptions().stride(2).padding(padding_mode_));

            // Store detail coefficients and update signal for next iteration
            coeffs.push_back(cD.squeeze());
            current_signal = cA.squeeze();
        }

        // --- 4. Finalize and Order Coefficients ---
        coeffs.push_back(current_signal); // The final approximation coefficient

        // Reverse to get the standard [cA_n, cD_n, ..., cD_1] order.
        std::reverse(coeffs.begin(), coeffs.end());

        return coeffs;
    }

} // namespace xt::transforms::signal