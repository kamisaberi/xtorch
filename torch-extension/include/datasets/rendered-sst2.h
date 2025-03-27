#pragma once
#include "../headers/datasets.h"
#include "base.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class RenderedSST2 : BaseDataset {
    public :
        RenderedSST2(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        RenderedSST2(const fs::path &root, DatasetArguments args);

    private:
        fs::path url = fs::path("https://openaipublic.azureedge.net/clip/data/rendered-sst2.tgz");
        fs::path dataset_file_name = "rendered-sst2.tgz";
        std::string dataset_file_md5 = "2384d08e9dcfa4bd55b324e610496ee5";
        fs::path dataset_folder_name = "rendered-sst2";
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);

    };
}
