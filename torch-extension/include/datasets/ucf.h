#pragma once
#include "../base/datasets.h"
#include "base.h"


namespace torch::ext::data::datasets {
   class UCF101 : public BaseDataset {
   public :
       UCF101(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

       UCF101(const fs::path &root, DatasetArguments args);
   private :
       void load_data(DataMode mode = DataMode::TRAIN);

       void check_resources(const std::string &root, bool download = false);


    };
}
