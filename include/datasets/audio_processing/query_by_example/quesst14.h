#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets {
    class QUESST14 : public xt::datasets::Dataset {
    public :
        explicit QUESST14(const std::string &root);

        QUESST14(const std::string &root, xt::datasets::DataMode mode);

        QUESST14(const std::string &root, xt::datasets::DataMode mode, bool download);

        QUESST14(const std::string &root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr <xt::Module> transformer);

        QUESST14(const std::string &root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr <xt::Module> transformer,
                   std::unique_ptr <xt::Module> target_transformer);

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
