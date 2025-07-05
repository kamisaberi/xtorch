#pragma once
#include "../common.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::datasets
{
    class PairedImageDataset :  torch::data::Dataset<PairedImageDataset>
    {
        PairedImageDataset (

        const std::string &input_dir,

        const std::string& target_dir
        );

        torch::data::Example<> get(size_t index) override;
        torch::optional<size_t> size() const override;

        std::vector<std::string> input_paths_, target_paths_;
    };
}

