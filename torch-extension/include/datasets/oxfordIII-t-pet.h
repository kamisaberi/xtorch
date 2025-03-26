#pragma once
#include "base.h"
#include "../headers/datasets.h"


namespace torch::ext::data::datasets {
    class OxfordIIITPet : BaseDataset {
    public :
        OxfordIIITPet(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        OxfordIIITPet(const fs::path &root, DatasetArguments args);

    private :
        vector<std::tuple<fs::path, std::string> > resources = {
            {
                fs::path("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"),
                "5c4f3ee8e5d25df40f4fd59a7f44e54c"
            },
            {
                fs::path("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"),
                "95a8c909bbe2e81eed6a22bccdf3f68f"
            }
        };
        vector<std::string> _VALID_TARGET_TYPES = {"category", "binary-category", "segmentation"};
        fs::path dataset_folder_name = "oxford-iii-pets";
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
