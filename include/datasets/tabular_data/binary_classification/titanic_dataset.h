#pragma once

#include "datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets
{
    class TitanicDataset : public xt::datasets::Dataset
    {
    public :
        explicit TitanicDataset(const std::string& root);
        TitanicDataset(const std::string& root, xt::datasets::DataMode mode);
        TitanicDataset(const std::string& root, xt::datasets::DataMode mode, bool download);
        TitanicDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        TitanicDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
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
