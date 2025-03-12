#pragma once

#include "../base/datasets.h"
#include "base.h"


using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
   class SEMEION : BaseDataset {
   public :
       SEMEION(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

       SEMEION(const fs::path &root, DatasetArguments args);

   private:
       fs::path url = fs::path("http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data");
       fs::path filename = fs::path("semeion.data");
       std::string md5_checksum = "cb545d371d2ce14ec121470795a77432";
       void load_data(DataMode mode = DataMode::TRAIN);

       void check_resources(const std::string &root, bool download = false);

    };
}
