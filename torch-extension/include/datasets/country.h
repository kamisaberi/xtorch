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
#include "../exceptions/implementation.h"


using namespace std;
namespace fs = std::filesystem;

namespace torch::ext::data::datasets {
   class Country211 : torch::data::Dataset<Country211> {
   private:

       std::string download_url = "https://openaipublic.azureedge.net/clip/data/";
       fs::path archive_file_name = "country211.tgz";
       std::string archive_file_md5 = "84988d7644798601126c29e9877aab6a";


    public :
       Country211();
    };
}
