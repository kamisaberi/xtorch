#pragma once

#include "../../common.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::datasets
{
    class Tedlium : public xt::datasets::Dataset
    {
    public :
        explicit Tedlium(const std::string& root);
        Tedlium(const std::string& root, xt::datasets::DataMode mode);
        Tedlium(const std::string& root, xt::datasets::DataMode mode, bool download);
        Tedlium(const std::string& root, xt::datasets::DataMode mode, bool download,
                std::unique_ptr<xt::Module> transformer);
        Tedlium(const std::string& root, xt::datasets::DataMode mode, bool download,
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
