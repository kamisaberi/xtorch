#pragma once

#include <torch/torch.h>
#include <torch/data/datasets.h>
#include <vector>
#include <string>
#include <sndfile.hh>
#include <fstream>
#include <sstream>

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {




    class AudioDataset : public torch::data::datasets::Dataset<AudioDataset> {
    public:
        AudioDataset(const std::string& audio_dir, const std::string& label_file) {
            std::ifstream label_stream(label_file);
            std::string line;
            while (std::getline(label_stream, line)) {
                std::istringstream iss(line);
                std::string filename, label_str;
                iss >> filename >> label_str;
                file_paths.push_back(audio_dir + "/" + filename + ".wav");
                labels.push_back(std::stoi(label_str));
            }
        }

        torch::data::Example<> get(size_t index) override {
            SndfileHandle file(file_paths[index]);
            std::vector<float> samples(file.frames());
            file.read(&samples[0], file.frames());
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            torch::Tensor waveform = torch::from_blob(samples.data(), {1, (int64_t)samples.size()}, options).clone();
            torch::Tensor label = torch::full({1}, labels[index], torch::kLong);
            return {waveform, label};
        }

        torch::optional<size_t> size() const override {
            return file_paths.size();
        }

    private:
        std::vector<std::string> file_paths;
        std::vector<int> labels;
    };


}
