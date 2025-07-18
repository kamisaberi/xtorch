
#include <datasets/general/audio_folder.h>

namespace xt::datasets {

    AudioFolder::AudioFolder(const std::string& audio_dir, const std::string& label_file)
    {
        std::ifstream label_stream(label_file);
        std::string line;
        while (std::getline(label_stream, line))
        {
            std::istringstream iss(line);
            std::string filename, label_str;
            iss >> filename >> label_str;
            files.push_back(audio_dir + "/" + filename + ".wav");
            labels.push_back(std::stoi(label_str));
        }
    }

    torch::data::Example<> AudioFolder::get(size_t index)
    {
        SndfileHandle file(files[index]);
        std::vector<float> samples(file.frames());
        file.read(&samples[0], file.frames());
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor waveform = torch::from_blob(samples.data(), {1, (int64_t)samples.size()}, options).clone();
        torch::Tensor label = torch::full({1}, labels[index], torch::kLong);
        return {waveform, label};
    }

    torch::optional<size_t> AudioFolder::size() const
    {
        return files.size();
    }





}
