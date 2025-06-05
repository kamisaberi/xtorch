#pragma once

#include "../../common.h"


using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets
{
    class Atari2600ALE : public xt::datasets::Dataset
    {
    public :
        explicit Atari2600ALE(const std::string& root);
        Atari2600ALE(const std::string& root, xt::datasets::DataMode mode);
        Atari2600ALE(const std::string& root, xt::datasets::DataMode mode, bool download);
        Atari2600ALE(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        Atari2600ALE(const std::string& root, xt::datasets::DataMode mode, bool download,
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
