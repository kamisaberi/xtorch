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
   class INaturalist : torch::data::Dataset<INaturalist> {
   private:
       std::map<string, std::tuple<fs::path, std::string>> resources = {
               {"2017",  {fs::path(
                       "https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz"),     "7c784ea5e424efaec655bd392f87301f"}},
               {"2018", {fs::path(
                       "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz"), "b1c6952ce38f31868cc50ea72d066cc3"}},
               {"2019", {fs::path(
                       "https://ml-inat-competition-datasets.s3.amazonaws.com/2019/train_val2019.tar.gz"), "c60a6e2962c9b8ccbd458d12c8582644"}},
               {"2021_train", {fs::path(
                       "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz"), "e0526d53c7f7b2e3167b2b43bb2690ed"}},
               {"2021_train_mini", {fs::path(
                       "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz"), "db6ed8330e634445efc8fec83ae81442"}},
               {"2021_valid", {fs::path(
                       "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz"), "f6f6e0e242e3d4c9569ba56400938afc"}}
       };



   public :
       INaturalist();
    };
}
