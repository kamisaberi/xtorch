#pragma once
#include "../headers/datasets.h"
#include "base.h"


namespace torch::ext::data::datasets {
    class CocoDetection : public BaseDataset {
    public :
        CocoDetection(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        CocoDetection(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

    class CocoCaptions : public BaseDataset {
    public :
        CocoCaptions(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        CocoCaptions(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
