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
namespace fs = std::filesystem

namespace torch::ext::data::datasets {
   class Country211 : torch::data::Dataset<Country211> {
   private:

       std::string download_url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
       fs::path archive_file_name = "cifar-10-binary.tar.gz";


    public :
       Country211();
    };
}
