#pragma once

#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
    [[deprecated("Flickr8k Dataset some files removed and Links are broken")]]
    class Flickr8k :public BaseDataset {
    public :
        Flickr8k(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Flickr8k(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

    [[deprecated("Flickr30k Dataset some files removed and Links are broken")]]
    class Flickr30k :public BaseDataset {
    public :
        Flickr30k(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Flickr30k(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
