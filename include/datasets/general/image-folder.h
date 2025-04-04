#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    enum class LabelsType {
        BY_FOLDER = 0,
        BY_NAME = 1,
    };


    class ImageFolder : public BaseDataset {
    public :

        ImageFolder(const std::string &root);
        ImageFolder(const std::string &root,bool load_sub_folders);
        ImageFolder(const std::string &root,bool load_sub_folders, DataMode mode);
        ImageFolder(const std::string &root,bool load_sub_folders, DataMode mode , LabelsType label_type);
        ImageFolder(const std::string &root,bool load_sub_folders, DataMode mode , LabelsType label_type, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private:
        vector<string> labels_name;
        LabelsType label_type = LabelsType::BY_FOLDER;
        bool load_sub_folders = false;
        void load_data();
    };
}
