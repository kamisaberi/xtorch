#pragma once
#include "datasets/common.h"



using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class EMNIST : public xt::datasets::Dataset {
    public :

        explicit EMNIST(const std::string& root);
        EMNIST(const std::string& root, xt::datasets::DataMode mode);
        EMNIST(const std::string& root, xt::datasets::DataMode mode, bool download);
        EMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
              std::unique_ptr<xt::Module> transformer);
        EMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
              std::unique_ptr<xt::Module> transformer,
              std::unique_ptr<xt::Module> target_transformer);


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
