#pragma once

#include "base.h"
#include "../base/datasets.h"


namespace torch::ext::data::datasets {
    class FlyingChairs : public BaseDataset {
    public :
        FlyingChairs(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        FlyingChairs(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

    class FlyingThings3D : public BaseDataset {
    public :
        FlyingThings3D(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        FlyingThings3D(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
