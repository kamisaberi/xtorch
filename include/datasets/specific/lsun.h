#pragma once
#include "base.h"
#include "../headers/datasets.h"


namespace xt::data::datasets {
    [[deprecated("LSUN Dataset files removed and Links are broken")]]
    class LSUN  : BaseDataset  {
    public :
        LSUN(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        LSUN(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
