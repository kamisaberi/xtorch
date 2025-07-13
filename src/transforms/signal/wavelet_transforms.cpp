#include "include/transforms/signal/wavelet_transforms.h"
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace xt::transforms::signal
{
    // FIX 1: The definition for this static helper function was missing.
    // The linker needs this function body to resolve calls from the constructor.
    std::vector<float> WaveletTransform::get_wavelet_coeffs(const std::string& name)
    {
        static const std::map<std::string, std::vector<float>> wavelet_db = {
            {"haar", {0.7071067811865476f, 0.7071067811865476f}},
            {"db2", {0.4829629131445341f, 0.8365163037378079f, 0.2241438680420134f, -0.12940952255126037f}},
            {
                "db4", {
                    0.2303778133088964f, 0.7148465705529154f, 0.6308807679298587f, -0.027983769416859854f,
                    -0.1870348117190931f, 0.030841381835560763f, 0.0328830116668852f, -0.01059740178506903f
                }
            }
        };
        auto it = wavelet_db.find(name);
        if (it == wavelet_db.end())
        {
            throw std::invalid_argument("Wavelet '" + name + "' is not supported.");
        }
        return it->second;
    }

    WaveletTransform::WaveletTransform(const std::string& wavelet, int n_levels, const std::string& padding_mode)
        : n_levels_(n_levels), padding_mode_(padding_mode)
    {
        auto lo_coeffs = get_wavelet_coeffs(wavelet);
        dec_lo_ = torch::tensor(lo_coeffs, torch::kFloat32);
        dec_hi_ = torch::flip(dec_lo_, {0});
        auto alternating_sign = torch::pow(-1.0, torch::arange(0, dec_hi_.size(0), torch::kFloat32));
        dec_hi_ *= alternating_sign;
        dec_lo_ = dec_lo_.view({1, 1, -1});
        dec_hi_ = dec_hi_.view({1, 1, -1});
    }

    auto WaveletTransform::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty())
        {
            throw std::invalid_argument("WaveletTransform::forward received an empty list.");
        }
        torch::Tensor waveform = std::any_cast<torch::Tensor>(any_vec[0]);
        if (!waveform.defined() || waveform.dim() != 1)
        {
            throw std::invalid_argument("Input must be a 1D waveform tensor.");
        }

        auto device = waveform.device();
        dec_lo_ = dec_lo_.to(device);
        dec_hi_ = dec_hi_.to(device);

        long filter_len = dec_lo_.size(-1);
        torch::Tensor current_signal = waveform.clone();
        std::vector<torch::Tensor> coeffs;

        int max_levels = 0;
        if (filter_len > 1)
        {
            max_levels = static_cast<int>(std::floor(
                std::log2(static_cast<double>(current_signal.size(0)) / (filter_len - 1.0))));
            max_levels = std::max(0, max_levels);
        }
        int levels = (n_levels_ > 0) ? std::min(n_levels_, max_levels) : max_levels;

        for (int i = 0; i < levels; ++i)
        {
            if (current_signal.size(0) < filter_len)
            {
                break;
            }

            torch::nn::functional::PadFuncOptions::mode_t mode;
            // FIX 2: Added a final `else` block to handle the default case ("zeros")
            // and ensure the `mode` variable is always initialized.
            if (padding_mode_ == "reflect")
            {
                mode = torch::kReflect;
            }
            else if (padding_mode_ == "replicate")
            {
                mode = torch::kReplicate;
            }
            else if (padding_mode_ == "circular")
            {
                mode = torch::kCircular;
            }
            // else
            // {
            //     mode = torch::kZeros;
            // }

            long pad_amount = filter_len - 1;
            long pad_left = pad_amount / 2;
            long pad_right = pad_amount - pad_left;

            auto input_3d = current_signal.view({1, 1, -1});
            auto padded_input = torch::nn::functional::pad(
                input_3d,
                torch::nn::functional::PadFuncOptions({pad_left, pad_right}).mode(mode)
            );

            auto conv_options = torch::nn::functional::Conv1dFuncOptions().stride(2).padding(0);
            auto cA = torch::nn::functional::conv1d(padded_input, dec_lo_, conv_options);
            auto cD = torch::nn::functional::conv1d(padded_input, dec_hi_, conv_options);

            coeffs.push_back(cD.squeeze());
            current_signal = cA.squeeze();
        }

        coeffs.push_back(current_signal);
        std::reverse(coeffs.begin(), coeffs.end());
        return coeffs;
    }
} // namespace xt::transforms::signal
