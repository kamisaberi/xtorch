#include <utils/audio.h>

#include <sndfile.h> // The header for libsndfile
#include <stdexcept>
#include <vector>
#include <iostream>

namespace xt::utils::audio {

    auto load(const std::string& path) -> std::pair<torch::Tensor, int> {
        // 1. Prepare to open the file
        SF_INFO sfinfo;
        // The sf_open function will fill sfinfo with the file's properties
        SNDFILE* sndfile = sf_open(path.c_str(), SFM_READ, &sfinfo);

        if (!sndfile) {
            // libsndfile provides a function to get a string for the last error
            throw std::runtime_error("Could not open audio file '" + path + "': " + sf_strerror(NULL));
        }

        // 2. Read the audio data
        // We read the entire file into a buffer. libsndfile will convert it to float for us.
        // The total number of float values is (number of frames) * (number of channels).
        std::vector<float> buffer(sfinfo.frames * sfinfo.channels);

        // sf_readf_float reads 'frames' of data.
        sf_count_t num_frames_read = sf_readf_float(sndfile, buffer.data(), sfinfo.frames);

        // Check if we read the expected number of frames
        if (num_frames_read != sfinfo.frames) {
            sf_close(sndfile);
            throw std::runtime_error("Error reading frames from audio file '" + path + "'.");
        }

        // 3. Close the file handle
        sf_close(sndfile);

        // 4. Convert the buffer to a LibTorch tensor
        // torch::from_blob is efficient but requires the data to stay in scope.
        // A copy with torch::tensor is safer and simpler.
        torch::Tensor tensor = torch::tensor(buffer);

        // Reshape the 1D buffer into a 2D tensor: (channels, frames)
        // Note: libsndfile reads data in an interleaved format [L1, R1, L2, R2, ...].
        // So we first create a (frames, channels) tensor and then transpose it.
        tensor = tensor.view({sfinfo.frames, sfinfo.channels}).transpose(0, 1);

        // 5. Convert to mono by averaging channels if necessary
        if (sfinfo.channels > 1) {
            tensor = tensor.mean(0); // Average across the channel dimension
        } else {
            tensor = tensor.squeeze(); // Remove the channel dimension
        }

        return {tensor, sfinfo.samplerate};
    }


    void save(const std::string& path, const torch::Tensor& tensor, int sample_rate) {
        // 1. Input Validation
        if (!tensor.defined() || tensor.numel() == 0) {
            throw std::runtime_error("Cannot save an un-defined or empty tensor.");
        }
        if (tensor.dim() < 1 || tensor.dim() > 2) {
            throw std::runtime_error("Input tensor must be 1D (mono) or 2D (channels, samples).");
        }
        if (tensor.dtype() != torch::kFloat32) {
            std::cerr << "Warning: Tensor is not float32. Casting to float32 for saving." << std::endl;
        }

        // 2. Prepare tensor for writing
        // Ensure the tensor is 2D (channels, samples) and on the CPU
        auto tensor_to_save = tensor.to(torch::kCPU, torch::kFloat32);
        if (tensor_to_save.dim() == 1) {
            tensor_to_save = tensor_to_save.unsqueeze(0); // Add a channel dimension
        }

        // libsndfile expects interleaved data (L1, R1, L2, R2, ...).
        // Our tensor is (channels, samples), so we transpose to (samples, channels)
        // and make it 'contiguous' in memory to get the right layout.
        tensor_to_save = tensor_to_save.transpose(0, 1).contiguous();

        // 3. Prepare SF_INFO for writing
        SF_INFO sfinfo;
        sfinfo.samplerate = sample_rate;
        sfinfo.channels = tensor_to_save.size(1);
        sfinfo.frames = tensor_to_save.size(0);
        // We'll save as 16-bit PCM WAV, a very common format.
        sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

        if (!sf_format_check(&sfinfo)) {
            throw std::runtime_error("Invalid format for saving audio file.");
        }

        // 4. Open the file for writing
        SNDFILE* sndfile = sf_open(path.c_str(), SFM_WRITE, &sfinfo);
        if (!sndfile) {
            throw std::runtime_error("Could not open file '" + path + "' for writing: " + sf_strerror(NULL));
        }

        // 5. Write the data
        // sf_writef_float expects a float pointer and automatically handles scaling to 16-bit.
        sf_count_t num_frames_written = sf_writef_float(sndfile, tensor_to_save.data_ptr<float>(), sfinfo.frames);

        if (num_frames_written != sfinfo.frames) {
            sf_close(sndfile);
            throw std::runtime_error("Error writing frames to audio file '" + path + "'.");
        }

        // 6. Finalize writing and close the file
        sf_write_sync(sndfile); // Ensure all data is written to disk
        sf_close(sndfile);
    }

} // namespace xt::utils::audio