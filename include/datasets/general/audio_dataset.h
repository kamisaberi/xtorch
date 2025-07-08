#pragma once

#include "../common.h"

#include <sndfile.hh>

using namespace std;
namespace fs = std::filesystem;


namespace xt::datasets
{
    class AudioDataset : public xt::datasets::Dataset
    {
    public:
        AudioDataset(const std::string& audio_dir, const std::string& label_file);

        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

    private:
        std::vector<std::string> files;
        std::vector<int> labels;
    };

}


