#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets
{
    class CocoDetection : public xt::datasets::Dataset
    {
    public :
        explicit CocoDetection(const std::string& root);
        CocoDetection(const std::string& root, xt::datasets::DataMode mode);
        CocoDetection(const std::string& root, xt::datasets::DataMode mode, bool download);
        CocoDetection(const std::string& root, xt::datasets::DataMode mode, bool download,
                      std::unique_ptr<xt::Module> transformer);
        CocoDetection(const std::string& root, xt::datasets::DataMode mode, bool download,
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
