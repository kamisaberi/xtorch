#pragma once

#include "../base/datasets.h"
#include "base.h"



namespace torch::ext::data::datasets {
   class ETH3DStereo : BaseDataset {

       ETH3DStereo(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);
       ETH3DStereo(const fs::path &root, DatasetArguments args);

    public :
       ETH3DStereo();
    };
}
