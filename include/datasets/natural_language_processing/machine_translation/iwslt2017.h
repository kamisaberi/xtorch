#pragma once

#include "../../common.h"


using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets
{
    class IWSLT2017 : public xt::datasets::Dataset
    {
    public :
        explicit IWSLT2017(const std::string& root);
        IWSLT2017(const std::string& root, xt::datasets::DataMode mode);
        IWSLT2017(const std::string& root, xt::datasets::DataMode mode, bool download);
        IWSLT2017(const std::string& root, xt::datasets::DataMode mode, bool download,
                  std::unique_ptr<xt::Module> transformer);
        IWSLT2017(const std::string& root, xt::datasets::DataMode mode, bool download,
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
