#pragma once

#include "../../common.h"


using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets
{
    class FSD50K : public xt::datasets::Dataset
    {
    public :
        explicit FSD50K(const std::string& root);
        FSD50K(const std::string& root, xt::datasets::DataMode mode);
        FSD50K(const std::string& root, xt::datasets::DataMode mode, bool download);
        FSD50K(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        FSD50K(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);

    private:

        // TODO fs::path dataset_folder_name
        fs::path dataset_folder_name = "?";

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void load_data();

        void check_resources();
    };
}
