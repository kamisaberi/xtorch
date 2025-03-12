#pragma once
#include "../base/datasets.h"
#include "base.h"


namespace torch::ext::data::datasets {
   class Sintel : BaseDataset {

   public :
       Sintel(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

       Sintel(const fs::path &root, DatasetArguments args);
   private :
       void load_data(DataMode mode = DataMode::TRAIN);

       void check_resources(const std::string &root, bool download = false);

   };
   class SintelStereo : torch::data::Dataset<SintelStereo> {

   public :
       SintelStereo(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

       SintelStereo(const fs::path &root, DatasetArguments args);
   private :
       void load_data(DataMode mode = DataMode::TRAIN);

       void check_resources(const std::string &root, bool download = false);
    };
}
