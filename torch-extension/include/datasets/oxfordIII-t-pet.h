#pragma once
#include "base.h"
#include "../headers/datasets.h"


namespace xt::data::datasets {
    class OxfordIIITPet : BaseDataset {
        /*
        """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

                - ``category`` (int): Label for one of the 37 pet categories.
                - ``binary-category`` (int): Binary label for cat or dog.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.

            If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

         */
    public :

        OxfordIIITPet(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        OxfordIIITPet(const fs::path &root, DatasetArguments args);

    private :
        vector<std::tuple<fs::path, std::string> > resources = {
            {
                fs::path("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"),
                "5c4f3ee8e5d25df40f4fd59a7f44e54c"
            },
            {
                fs::path("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"),
                "95a8c909bbe2e81eed6a22bccdf3f68f"
            }
        };
        vector<std::string> _VALID_TARGET_TYPES = {"category", "binary-category", "segmentation"};
        fs::path dataset_folder_name = "oxford-iii-pets";
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
