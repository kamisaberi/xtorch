#pragma once

#include "../base/datasets.h"
#include "base.h"


namespace torch::ext::data::datasets {
    class Flickr8k :public BaseDataset {
    public :
        Flickr8k(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Flickr8k(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

    class Flickr30k :public BaseDataset {
    public :
        Flickr30k(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Flickr30k(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
