#pragma once
#include <headers/datasets.h>
#include "datasets/base/mnist_base.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {


    class EMNIST : public MNISTBase {
    public :
        EMNIST(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        EMNIST(const fs::path &root, DatasetArguments args);

    private:
        std::string url = "https://biometrics.nist.gov/cs_links/EMNIST/";
        fs::path dataset_folder_name = "EMNIST/raw";
        fs::path archive_file_name = "gzip.zip";
        std::string archive_file_md5 = "58c8d27c78d21e728a6bc7b3cc06412e";

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

}
