#pragma once

#include "include/datasets/common.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::datasets
{
    class Caltech256 : public xt::datasets::Dataset
    {
    public :
        explicit Caltech256(const std::string& root);
        Caltech256(const std::string& root, xt::datasets::DataMode mode);
        Caltech256(const std::string& root, xt::datasets::DataMode mode, bool download);
        Caltech256(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        Caltech256(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);

    private:
        std::tuple<string, string, string> resources = {
            "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1",
            "256_ObjectCategories.tar",
            "67b4f42ca05d46448c6bb8ecd2220f6d"
        };

        fs::path dataset_folder_name = "256_ObjectCategories";

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void load_data();

        void check_resources();
    };
}
