#pragma once

#include "datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {


    class CREStereo : public xt::datasets::Dataset {
    public :
        explicit CREStereo(const std::string& root);
        CREStereo(const std::string& root, xt::datasets::DataMode mode);
        CREStereo(const std::string& root, xt::datasets::DataMode mode, bool download);
        CREStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        CREStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
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
