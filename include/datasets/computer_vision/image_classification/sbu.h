#pragma once

#include "datasets/common.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class SBU : xt::datasets::Dataset {
        /*
        """`SBU Captioned Photo <http://www.cs.virginia.edu/~vicente/sbucaptions/>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where tarball
            ``SBUCaptionedPhotoDataset.tar.gz`` exists.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

         */
    public :
        explicit SBU(const std::string& root);
        SBU(const std::string& root, xt::datasets::DataMode mode);
        SBU(const std::string& root, xt::datasets::DataMode mode, bool download);
        SBU(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        SBU(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);


    private :
        fs::path url = fs::path("https://www.cs.rice.edu/~vo9/sbucaptions/SBUCaptionedPhotoDataset.tar.gz");
        fs::path dataset_file_name = fs::path("SBUCaptionedPhotoDataset.tar.gz");
        std::string dataset_file_md5 = "9aec147b3488753cf758b4d493422285";
        fs::path dataset_folder_name = "sbu";

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void load_data();

        void check_resources();



    };
}
