#pragma once

#include "../base/datasets.h"
#include "base.h"


namespace torch::ext::data::datasets {
   class VOCSegmentation : BaseDataset {

   public :
       VOCSegmentation(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

       VOCSegmentation(const fs::path &root, DatasetArguments args);
   private :
       void load_data(DataMode mode = DataMode::TRAIN);

       void check_resources(const std::string &root, bool download = false);

   };
   class VOCDetection : BaseDataset {

   public :
       VOCDetection(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

       VOCDetection(const fs::path &root, DatasetArguments args);
   private :
       void load_data(DataMode mode = DataMode::TRAIN);

       void check_resources(const std::string &root, bool download = false);
    };
}
