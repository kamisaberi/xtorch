#pragma once


#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {



    class TensorDataset : public BaseDataset {
    public :
        TensorDataset(const std::string &file_path);
        TensorDataset(const std::string &file_path,DataMode mode);
        TensorDataset(const std::string &file_path,DataMode mode , vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };





}
