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
    class RenderedSST2 : torch::data::Dataset<RenderedSST2> {
    private:
        fs::path url = fs::path("https://openaipublic.azureedge.net/clip/data/rendered-sst2.tgz");
        std::string md5 = "2384d08e9dcfa4bd55b324e610496ee5";

    public :
        RenderedSST2();
    };
}
