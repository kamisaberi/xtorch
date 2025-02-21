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
    class WIDERFace : torch::data::Dataset<WIDERFace> {
    private:
        fs::path BASE_FOLDER = fs::path("widerface");
        std::vector<std::tuple<std::string, std::string, fs::path> > FILE_LIST = {
                {"15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M", "3fedf70df600953d25982bcd13d91ba2", fs::path("WIDER_train.zip")},
                {"1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q", "dfa7d7e790efa35df3788964cf0bbaea", fs::path("WIDER_val.zip")},
                {"1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T", "e5d8f4248ed24c334bbd12f49c29dd40", fs::path("WIDER_test.zip")}
        };
        std::tuple<fs::path, std::string, fs::path> ANNOTATIONS_FILE = {
                fs::path("http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"),
                "0e3767bcf0e326556d407bf5bff5d27c",
                fs::path("wider_face_split.zip")
        };

    public :
        WIDERFace();
    };
}
