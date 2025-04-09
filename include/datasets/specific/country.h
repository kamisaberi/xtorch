#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {
    class Country211 : public BaseDataset {
    public:
        explicit  Country211(const std::string &root);
        Country211(const std::string &root, DataMode mode);
        Country211(const std::string &root, DataMode mode , bool download);
        Country211(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private:
        std::string url = "https://openaipublic.azureedge.net/clip/data/";
        fs::path dataset_file_name = "country211.tgz";
        std::string dataset_file_md5 = "84988d7644798601126c29e9877aab6a";
        fs::path dataset_folder_name = "country211";

    };
}
