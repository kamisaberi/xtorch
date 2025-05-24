#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets
{
    class AgNews : public xt::datasets::Dataset
    {
    public :
        explicit AgNews(const std::string& root);
        AgNews(const std::string& root, xt::datasets::DataMode mode);
        AgNews(const std::string& root, xt::datasets::DataMode mode, bool download);
        AgNews(const std::string& root, xt::datasets::DataMode mode, bool download,
               std::unique_ptr<xt::Module> transformer);
        AgNews(const std::string& root, xt::datasets::DataMode mode, bool download,
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
