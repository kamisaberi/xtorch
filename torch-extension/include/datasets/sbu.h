#pragma once
#include "../headers/datasets.h"
#include "base.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class SBU : BaseDataset {
    public :
        SBU(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        SBU(const fs::path &root, DatasetArguments args);

    private :
        fs::path url = fs::path("https://www.cs.rice.edu/~vo9/sbucaptions/SBUCaptionedPhotoDataset.tar.gz");
        fs::path dataset_file_name = fs::path("SBUCaptionedPhotoDataset.tar.gz");
        std::string dataset_file_md5 = "9aec147b3488753cf758b4d493422285";
        fs::path dataset_folder_name = "sbu";
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);


    };
}
