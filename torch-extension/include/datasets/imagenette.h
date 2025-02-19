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
    class Imagenette : torch::data::Dataset<Imagenette> {
    private:
        std::map<string, std::tuple<fs::path, std::string>> resources = {
                {"full",  {fs::path(
                        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"),     "fe2fc210e6bb7c5664d602c3cd71e612"}},
                {"320px", {fs::path(
                        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"), "3df6f0d01a2c9592104656642f5e78a3"}},
                {"160px", {fs::path(
                        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"), "e793b78cc4c9e9a4ccc0c1155377a412"}}
        };


    public :
        Imagenette();
    };
}
