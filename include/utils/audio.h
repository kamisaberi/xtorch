#pragma once

#include <string>
#include <utility> // For std::pair
#include <torch/torch.h>

namespace xt::utils::audio {

    /**
     * @brief Loads an audio file from the specified path.
     *
     * This function reads an audio file (e.g., WAV, FLAC) and returns its
     * contents as a mono, 32-bit floating-point LibTorch tensor, along with
     * the sample rate. If the audio file has multiple channels, they are
     * averaged to produce a mono signal. The tensor values are normalized
     * to the range [-1.0, 1.0].
     *
     * @param path The file path to the audio file.
     * @return A std::pair containing:
     *         - torch::Tensor: A 1D tensor of the audio data (mono, float32).
     *         - int: The sample rate of the audio file in Hz.
     * @throws std::runtime_error if the file cannot be opened or read.
     */
    auto load(const std::string& path) -> std::pair<torch::Tensor, int>;

    /**
     * @brief Saves a LibTorch tensor to an audio file.
     *
     * This function writes a 1D (mono) or 2D (channels, samples) float tensor
     * to a WAV file with 16-bit PCM encoding. The tensor is assumed to be in
     * the range [-1.0, 1.0].
     *
     * @param path The file path where the audio will be saved.
     * @param tensor The audio data to save. Should be a 1D or 2D float tensor.
     * @param sample_rate The sample rate of the audio in Hz.
     * @throws std::runtime_error if the tensor is invalid or the file cannot be written.
     */
    void save(const std::string& path, const torch::Tensor& tensor, int sample_rate);

} // namespace xt::utils::audio