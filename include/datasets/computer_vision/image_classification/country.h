#pragma once

#include "datasets/common.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class Country211 : public xt::datasets::Dataset {
    public:
        explicit Country211(const std::string& root);
        Country211(const std::string& root, xt::datasets::DataMode mode);
        Country211(const std::string& root, xt::datasets::DataMode mode, bool download);
        Country211(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        Country211(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);


    private:
        std::string url = "https://openaipublic.azureedge.net/clip/data/";
        fs::path dataset_file_name = "country211.tgz";
        std::string dataset_file_md5 = "84988d7644798601126c29e9877aab6a";
        fs::path dataset_folder_name = "country211";


        bool download = false;
        fs::path root;
        fs::path dataset_path;


        void load_data();

        void check_resources();


    };
}
