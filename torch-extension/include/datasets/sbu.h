#pragma once
#include "../base/datasets.h"
#include "base.h"


using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
    class SBU : BaseDataset {
    public :
        SBU(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        SBU(const fs::path &root, DatasetArguments args);

    private :
        fs::path url = fs::path("https://www.cs.rice.edu/~vo9/sbucaptions/SBUCaptionedPhotoDataset.tar.gz");
        fs::path filename = fs::path("SBUCaptionedPhotoDataset.tar.gz");
        std::string md5_checksum = "9aec147b3488753cf758b4d493422285";
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);


    };
}
