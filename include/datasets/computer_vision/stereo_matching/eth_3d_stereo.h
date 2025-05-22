#pragma once

#include "datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {
    class ETH3DStereo : public xt::datasets::Dataset {
    public:

        explicit ETH3DStereo(const std::string& root);
        ETH3DStereo(const std::string& root, xt::datasets::DataMode mode);
        ETH3DStereo(const std::string& root, xt::datasets::DataMode mode, bool download);
        ETH3DStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        ETH3DStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);



    private :
        // TODO fs::path dataset_folder_name
        fs::path dataset_folder_name = "?";

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void load_data();

        void check_resources();
    };
}
