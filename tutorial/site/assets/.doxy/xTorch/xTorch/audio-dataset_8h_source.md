

# File audio-dataset.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**general**](dir_3e490c73b2bbc01f3b90ef3b6e284c64.md) **>** [**audio-dataset.h**](audio-dataset_8h.md)

[Go to the documentation of this file](audio-dataset_8h.md)


```C++
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

    class StackedAudioDataset : public BaseDataset {
        public :
            StackedAudioDataset(const std::string &folder_path);

        StackedAudioDataset(const std::string &folder_path, DataMode mode);

        StackedAudioDataset(const std::string &folder_path, DataMode mode, bool load_sub_folders);

        StackedAudioDataset(const std::string &folder_path, DataMode mode, bool load_sub_folders,
                          vector<std::function<torch::Tensor(torch::Tensor)> > transforms);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;

        void load_data();
    };


}
```


