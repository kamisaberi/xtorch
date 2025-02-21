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
    class SBU : torch::data::Dataset<SBU> {
    private :
        fs::path url = fs::path("https://www.cs.rice.edu/~vo9/sbucaptions/SBUCaptionedPhotoDataset.tar.gz");
        fs::path filename = fs::path("SBUCaptionedPhotoDataset.tar.gz");
        std::string md5_checksum = "9aec147b3488753cf758b4d493422285";


    public :
        SBU();
    };
}
