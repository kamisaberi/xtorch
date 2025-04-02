#pragma once
#include "base.h"
#include "../headers/datasets.h"


using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {
    class INaturalist : BaseDataset {
        /*
        """`iNaturalist <https://github.com/visipedia/inat_comp>`_ Dataset.
        Args:
            root (str or ``pathlib.Path``): Root directory of dataset where the image files are stored.
                This class does not require/use annotation files.
            version (string, optional): Which version of the dataset to download/use. One of
                '2017', '2018', '2019', '2021_train', '2021_train_mini', '2021_valid'.
                Default: `2021_train`.
            target_type (string or list, optional): Type of target to use, for 2021 versions, one of:

                - ``full``: the full category (species)
                - ``kingdom``: e.g. "Animalia"
                - ``phylum``: e.g. "Arthropoda"
                - ``class``: e.g. "Insecta"
                - ``order``: e.g. "Coleoptera"
                - ``family``: e.g. "Cleridae"
                - ``genus``: e.g. "Trichodes"

                for 2017-2019 versions, one of:

                - ``full``: the full (numeric) category
                - ``super``: the super category, e.g. "Amphibians"

                Can also be a list to output a tuple with all specified target types.
                Defaults to ``full``.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
        """
         */
    public :
        INaturalist(const std::string &root);
        INaturalist(const std::string &root, DataMode mode);
        INaturalist(const std::string &root, DataMode mode , bool download);
        INaturalist(const std::string &root, DataMode mode , bool download, TransformType transforms);

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

        void load_data();

        void check_resources();
    };
}
