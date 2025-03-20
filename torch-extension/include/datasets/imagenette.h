#pragma once
#include "base.h"
#include "../base/datasets.h"


using namespace std;
namespace fs = std::filesystem;

namespace torch::ext::data::datasets {
    class Imagenette : BaseDataset {
    public :
        Imagenette(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Imagenette(const fs::path &root, DatasetArguments args);

    private:
        std::map<string, std::tuple<fs::path, std::string> > resources = {
            {
                "full", {
                    fs::path("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"),
                    "fe2fc210e6bb7c5664d602c3cd71e612"
                }
            },
            {
                "320px", {
                    fs::path("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"),
                    "3df6f0d01a2c9592104656642f5e78a3"
                }
            },
            {
                "160px", {
                    fs::path("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"),
                    "e793b78cc4c9e9a4ccc0c1155377a412"
                }
            }
        };
        fs::path dataset_folder_name = "imagenette";

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
