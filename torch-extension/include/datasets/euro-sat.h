#pragma once

#include "../headers/datasets.h"
#include "base.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class EuroSAT :public BaseDataset {
    public :
        EuroSAT(const std::string &root);
        EuroSAT(const std::string &root, DataMode mode);
        EuroSAT(const std::string &root, DataMode mode , bool download);
        EuroSAT(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);


    private:
        std::string url =
                "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip";
        fs::path archive_file_name = "EuroSAT.zip";
        std::string archive_file_md5 = "c8fa014336c82ac7804f0398fcb19387";
        fs::path dataset_folder_name = "euro-sat";

        void load_data();

        void check_resources();
    };
}
