#pragma once

#include "../base/datasets.h"
#include "base.h"


namespace torch::ext::data::datasets {
    class FakeData :public BaseDataset {
    public :
        FakeData(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        FakeData(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
