#pragma once

#include "datasets/common.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class EuroSAT :public  xt::datasets::Dataset {
    public :
        explicit EuroSAT(const std::string& root);
        EuroSAT(const std::string& root, xt::datasets::DataMode mode);
        EuroSAT(const std::string& root, xt::datasets::DataMode mode, bool download);
        EuroSAT(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        EuroSAT(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);


    private:
        std::string url =
                "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip";
        fs::path archive_file_name = "EuroSAT.zip";
        std::string archive_file_md5 = "c8fa014336c82ac7804f0398fcb19387";
        fs::path dataset_folder_name = "euro-sat";

        bool download = false;
        fs::path root;
        fs::path dataset_path;


        void load_data();

        void check_resources();
    };
}
