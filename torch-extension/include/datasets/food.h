#pragma once

//#include <vector>
#include <fstream>
//#include <iostream>
//#include <string>
#include <filesystem>
//#include <curl/curl.h>
#include <torch/torch.h>
//#include <vector>
//#include <fstream>
//#include <iostream>
//#include <string>
//#include <filesystem>
#include "../utils/downloader.h"
#include "../utils/archiver.h"
#include "../utils/md5.h"

using namespace std;
namespace fs = std::filesystem;

namespace torch::ext::data::datasets {
   class Food101 : torch::data::Dataset<Food101> {
   private :
       std::string download_url  = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz";
       fs::path archive_file_name = "food-101.tar.gz";
       std::string archive_file_md5 = "85eeb15f3717b99a5da872d97d918f87"


   public :
       Food101();
    };
}
