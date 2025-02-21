#pragma once

//#include <vector>
#include <fstream>
//#include <iostream>
//#include <string>
#include <filesystem>
//#include <curl/curl.h>
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <map>
//#include <fstream>
//#include <iostream>
//#include <string>
//#include <filesystem>
#include "../utils/downloader.h"
#include "../utils/archiver.h"
#include "../utils/md5.h"
#include "../exceptions/implementation.h"

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
