#pragma once

#include "base.h"
#include "../base/datasets.h"


namespace torch::ext::data::datasets {
    enum class LabelsType {
        BY_FOLDER = 0,
        BY_NAME = 1,
    };


    class ImageFolder : public BaseDataset {
    public :
        ImageFolder(const std::string &root, DataMode mode = DataMode::TRAIN, LabelsType label_type = LabelsType::BY_FOLDER);

        ImageFolder(const fs::path &root, DatasetArguments args);

    private:
        fs::path dataset_folder_name = "imagenette";
        vector<string> labels_name;
        LabelsType label_type = LabelsType::BY_FOLDER;

        void load_data();

        void check_resources(const std::string &root, bool download = false);
    };
}
