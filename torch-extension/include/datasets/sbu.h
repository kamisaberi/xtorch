#pragma once
#include "../base/datasets.h"


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
