#pragma once

#include "../../common.h"


using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets {
    class LIBRITTS : public xt::datasets::Dataset {
    public :
        explicit LIBRITTS(const std::string &root);

        LIBRITTS(const std::string &root, xt::datasets::DataMode mode);

        LIBRITTS(const std::string &root, xt::datasets::DataMode mode, bool download);

        LIBRITTS(const std::string &root, xt::datasets::DataMode mode, bool download,
                 std::unique_ptr <xt::Module> transformer);

        LIBRITTS(const std::string &root, xt::datasets::DataMode mode, bool download,
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
