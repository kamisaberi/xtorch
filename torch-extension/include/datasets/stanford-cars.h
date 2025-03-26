#pragma once
#include "../headers/datasets.h"
#include "base.h"

namespace torch::ext::data::datasets {
   class StanfordCars : BaseDataset {

   public :
       StanfordCars(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

       StanfordCars(const fs::path &root, DatasetArguments args);
   private :
       void load_data(DataMode mode = DataMode::TRAIN);

       void check_resources(const std::string &root, bool download = false);

   };
}
