#pragma once
#include "base.h"
#include "../headers/datasets.h"


using namespace std;
namespace fs = std::filesystem;

namespace torch::ext::data::datasets {
    class INaturalist : BaseDataset {
    public :
        INaturalist(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        INaturalist(const fs::path &root, DatasetArguments args);

    private:
        std::map<string, std::tuple<fs::path, std::string> > resources = {
            {
                "2017", {
                    fs::path(
                        "https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz"),
                    "7c784ea5e424efaec655bd392f87301f"
                }
            },
            {
                "2018", {
                    fs::path(
                        "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz"),
                    "b1c6952ce38f31868cc50ea72d066cc3"
                }
            },
            {
                "2019", {
                    fs::path(
                        "https://ml-inat-competition-datasets.s3.amazonaws.com/2019/train_val2019.tar.gz"),
                    "c60a6e2962c9b8ccbd458d12c8582644"
                }
            },
            {
                "2021_train", {
                    fs::path(
                        "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz"),
                    "e0526d53c7f7b2e3167b2b43bb2690ed"
                }
            },
            {
                "2021_train_mini", {
                    fs::path(
                        "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz"),
                    "db6ed8330e634445efc8fec83ae81442"
                }
            },
            {
                "2021_valid", {
                    fs::path(
                        "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz"),
                    "f6f6e0e242e3d4c9569ba56400938afc"
                }
            }
        };

        fs::path dataset_folder_name = "inaturalist";

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
