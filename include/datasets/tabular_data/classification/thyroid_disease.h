#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets
{
    class ThyroidDisease : public xt::datasets::Dataset
    {
    public :
        explicit ThyroidDisease(const std::string& root);
        ThyroidDisease(const std::string& root, xt::datasets::DataMode mode);
        ThyroidDisease(const std::string& root, xt::datasets::DataMode mode, bool download);
        ThyroidDisease(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        ThyroidDisease(const std::string& root, xt::datasets::DataMode mode, bool download,
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
