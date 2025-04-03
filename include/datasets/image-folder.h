#pragma once

#include "base.h"
#include "../headers/datasets.h"


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

        // ImageFolder(const std::string &root,bool load_sub_folders , DataMode mode , LabelsType label_type = LabelsType::BY_FOLDER );

        // ImageFolder(const fs::path &root, DatasetArguments args);

    private:
        fs::path dataset_folder_name = "imagenette";
        vector<string> labels_name;
        LabelsType label_type = LabelsType::BY_FOLDER;
        bool load_sub_folders = false;

        void load_data();

        // void check_resources(const std::string &root, bool download = false);
    };
}
