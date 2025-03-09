#pragma once
#include "../base/datasets.h"
#include "base.h"


using namespace std;
namespace fs = std::filesystem;

namespace torch::ext::data::datasets {
    class Country211 : BaseDataset {
    private:
        std::string download_url = "https://openaipublic.azureedge.net/clip/data/";
        fs::path archive_file_name = "country211.tgz";
        std::string archive_file_md5 = "84988d7644798601126c29e9877aab6a";

    public :
        Country211();
    };
}
