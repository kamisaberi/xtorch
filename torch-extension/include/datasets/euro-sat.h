#pragma once

#include "../base/datasets.h"
#include "base.h"

using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
    class EuroSAT : BaseDataset {
        public :
    EuroSAT(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);
        EuroSAT(const fs::path &root, DatasetArguments args);

    private:
        std::string url =
                "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip";
        fs::path archive_file_name = "EuroSAT.zip";
        std::string archive_file_md5 = "c8fa014336c82ac7804f0398fcb19387";
        fs::path dataset_folder_name = "euro-sat";
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);

    };
}
