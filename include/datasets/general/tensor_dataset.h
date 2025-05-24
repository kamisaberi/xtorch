#pragma once


#include "include/datasets/common.h"

namespace xt::datasets
{
    class TensorDataset : public xt::datasets::Dataset
    {
    public :
        TensorDataset(const std::string& file_path);
        TensorDataset(const std::string& file_path, DataMode mode);
        TensorDataset(const std::string& file_path, DataMode mode, std::unique_ptr<xt::Module> target_transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };
}
