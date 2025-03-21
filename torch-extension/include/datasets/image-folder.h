#pragma once

#include "base.h"
#include "../base/datasets.h"



namespace torch::ext::data::datasets {
    class ImageFolder : public BaseDataset {
        public :
            ImageFolder(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);


        ImageFolder(const fs::path &root, DatasetArguments args);

    private:
        fs::path dataset_folder_name = "imagenette";
        vector<string> labels_name;
        void load_data(DataMode mode = DataMode::TRAIN);
        void check_resources(const std::string &root, bool download = false);

    };




}

