#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets {
    class GTSRB : xt::datasets::Dataset {
        /*
        """`German Traffic Sign Recognition Benchmark (GTSRB) <https://benchmark.ini.rub.de/>`_ Dataset.

        Args:
            root (str or ``pathlib.Path``): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
                version. E.g, ``transforms.RandomCrop``.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            download (bool, optional): If True, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
        """
         */
    public :
        explicit GTSRB(const std::string& root);
        GTSRB(const std::string& root, xt::datasets::DataMode mode);
        GTSRB(const std::string& root, xt::datasets::DataMode mode, bool download);
        GTSRB(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        GTSRB(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);

    private :
        // base_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
        //
        // if self._split == "train":
        //     download_and_extract_archive(
        //         f"{base_url}GTSRB-Training_fixed.zip",
        //         download_root=str(self._base_folder),
        //         md5="513f3c79a4c5141765e10e952eaa2478",
        //     )
        // else:
        //     download_and_extract_archive(
        //         f"{base_url}GTSRB_Final_Test_Images.zip",
        //         download_root=str(self._base_folder),
        //         md5="c7e4e6327067d32654124b0fe9e82185",
        //     )
        //     download_and_extract_archive(
        //         f"{base_url}GTSRB_Final_Test_GT.zip",
        //         download_root=str(self._base_folder),
        //         md5="fe31e9c9270bbcd7b84b7f21a9d9d9e5",
        //     )

        // TODO fs::path dataset_folder_name
        fs::path dataset_folder_name = "?";

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void load_data();

        void check_resources();
    };
}
