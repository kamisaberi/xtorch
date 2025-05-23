#pragma once
#include "datasets/common.h"



using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class EMNIST : public xt::datasets::Dataset {
    public :
        EMNIST(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        EMNIST(const fs::path &root);

    private:
        std::string url = "https://biometrics.nist.gov/cs_links/EMNIST/";
        fs::path dataset_folder_name = "EMNIST/raw";
        fs::path archive_file_name = "gzip.zip";
        std::string archive_file_md5 = "58c8d27c78d21e728a6bc7b3cc06412e";

        void load_data();

        void check_resources();

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void read_images(const std::string& file_path, int num_images);
        void read_labels(const std::string& file_path, int num_labels);


    };

}
