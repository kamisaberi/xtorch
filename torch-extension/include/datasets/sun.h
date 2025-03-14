#pragma once
#include "../base/datasets.h"
#include "base.h"


using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
    class SUN397 : BaseDataset {
    public :
        SUN397(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        SUN397(const fs::path &root, DatasetArguments args);

    private :
        fs::path url = fs::path("http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz");
        fs::path dataset_file_name = "SUN397.tar.gz";
        std::string dataset_file_md5 = "8ca2778205c41d23104230ba66911c7a";
        fs::path dataset_folder_name = "sun397";

        void load_data(DataMode mode = DataMode::TRAIN);


        void check_resources(const std::string &root, bool download = false);

    };
}
