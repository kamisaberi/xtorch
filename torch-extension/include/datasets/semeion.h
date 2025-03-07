#pragma once

#include "../base/datasets.h"


using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
   class SEMEION : torch::data::Dataset<SEMEION> {
   private:
       fs::path url = fs::path("http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data");
       fs::path filename = fs::path("semeion.data");
       std::string md5_checksum = "cb545d371d2ce14ec121470795a77432";

   public :
       SEMEION();
    };
}
