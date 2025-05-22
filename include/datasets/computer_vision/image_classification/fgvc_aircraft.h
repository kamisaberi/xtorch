#pragma once

#include "datasets/common.h"


using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {
    class FGVCAircraft :public xt::datasets::Dataset {
    public :
        explicit FGVCAircraft(const std::string& root);
        FGVCAircraft(const std::string& root, xt::datasets::DataMode mode);
        FGVCAircraft(const std::string& root, xt::datasets::DataMode mode, bool download);
        FGVCAircraft(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        FGVCAircraft(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);



    private :
        fs::path url = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz";
        std::string dataset_file_md5 = "85eeb15f3717b99a5da872d97d918f87";
        std::string dataset_file_name = "fgvc-aircraft-2013b.tar.gz";
        fs::path dataset_folder_name = "fgvc-aircraft-2013b";

        bool download = false;
        fs::path root;
        fs::path dataset_path;


        void load_data();

        void check_resources();

    };
}
