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
    class SUN397 : torch::data::Dataset<SUN397> {
    private :
        fs::path url = fs::path("http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz");
        std::string md5 = "8ca2778205c41d23104230ba66911c7a";

    public :
        SUN397();
    };
}
