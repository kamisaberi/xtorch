#pragma once

#include "datasets/common.h"

namespace xt::data::datasets {
    enum class LabelsType {
        BY_FOLDER = 0,
        BY_NAME = 1,
    };


    class ImageFolder : public xt::datasets::Dataset {
    public :

        ImageFolder(const std::string &root);
        ImageFolder(const std::string &root,bool load_sub_folders);
        ImageFolder(const std::string &root,bool load_sub_folders, DataMode mode);
        ImageFolder(const std::string &root,bool load_sub_folders, DataMode mode , LabelsType label_type);
        ImageFolder(const std::string &root,bool load_sub_folders, DataMode mode , LabelsType label_type, std::unique_ptr<xt::Module> transformer);

    private:
        vector<string> labels_name;
        LabelsType label_type = LabelsType::BY_FOLDER;
        bool load_sub_folders = false;
        void load_data();

        bool download = false;
        fs::path root;
        fs::path dataset_path;


    };
}
