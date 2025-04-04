#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {
    enum class ImageType {
        FULL = 0,
        PX320 = 1,
        PX160 = 2
    };

    const std::unordered_map<ImageType, std::string> ImageTypeToString = {
        {ImageType::FULL, "full"},
        {ImageType::PX320, "320px"},
        {ImageType::PX160, "160px"}
    };

    inline std::string getImageTypeValue(ImageType type) {
        auto it = ImageTypeToString.find(type);
        if (it != ImageTypeToString.end()) {
            return it->second;
        }
        return "unknown"; // Default value if enum not found
    }

    class Imagenette : public BaseDataset {
    public :
        Imagenette(const std::string &root);
        Imagenette(const std::string &root, DataMode mode);
        Imagenette(const std::string &root, DataMode mode , bool download);
        Imagenette(const std::string &root, DataMode mode , bool download, ImageType type );
        Imagenette(const std::string &root, DataMode mode , bool download, ImageType type , TransformType transforms);

    private:
        std::map<string, std::tuple<fs::path, fs::path, fs::path, std::string> > resources = {
            {
                "full", {
                    fs::path("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"),
                    fs::path("imagenette2.tgz"),
                    fs::path("imagenette2"),
                    "fe2fc210e6bb7c5664d602c3cd71e612"
                }
            },
            {
                "320px", {
                    fs::path("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"),
                    fs::path("imagenette2-320.tgz"),
                    fs::path("imagenette2-320"),
                    "3df6f0d01a2c9592104656642f5e78a3"
                }
            },
            {
                "160px", {
                    fs::path("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"),
                    fs::path("imagenette2-160.tgz"),
                    fs::path("imagenette2-160"),
                    "e793b78cc4c9e9a4ccc0c1155377a412"
                }
            }
        };
        ImageType type = ImageType::PX160;
        fs::path dataset_folder_name = "imagenette";
        vector<string> labels_name;

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
