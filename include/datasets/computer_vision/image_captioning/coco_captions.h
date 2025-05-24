#pragma once

#include "include/datasets/common.h"


namespace xt::datasets {
    class CocoCaptions : public xt::datasets::Dataset {
    public :
        explicit CocoCaptions(const std::string &root);

        CocoCaptions(const std::string &root, xt::datasets::DataMode mode);

        CocoCaptions(const std::string &root, xt::datasets::DataMode mode, bool download);

        CocoCaptions(const std::string &root, xt::datasets::DataMode mode, bool download,
                     std::unique_ptr <xt::Module> transformer);

        CocoCaptions(const std::string &root, xt::datasets::DataMode mode, bool download,
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
