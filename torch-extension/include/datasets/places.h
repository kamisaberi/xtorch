#pragma once
#include "../headers/datasets.h"
#include "base.h"



namespace xt::data::datasets {
   class Places365 : BaseDataset {
       /*
       """`Places365 <http://places2.csail.mit.edu/index.html>`_ classification dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of the Places365 dataset.
        split (string, optional): The dataset split. Can be one of ``train-standard`` (default), ``train-challenge``,
            ``val``.
        small (bool, optional): If ``True``, uses the small images, i.e. resized to 256 x 256 pixels, instead of the
            high resolution ones.
        download (bool, optional): If ``True``, downloads the dataset components and places them in ``root``. Already
            downloaded archives are not downloaded again.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    Raises:
        RuntimeError: If ``download is False`` and the meta files, i.e. the devkit, are not present or corrupted.
        RuntimeError: If ``download is True`` and the image archive is already extracted.
    """

        */
   public :
       Places365(const std::string &root);
       Places365(const std::string &root, DataMode mode);
       Places365(const std::string &root, DataMode mode , bool download);
       Places365(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

   private :

//        _SPLITS = ("train-standard", "train-challenge", "val")
//        _BASE_URL = "http://data.csail.mit.edu/places/places365/"
//        # {variant: (archive, md5)}
//        _DEVKIT_META = {
//        "standard": ("filelist_places365-standard.tar", "35a0585fee1fa656440f3ab298f8479c"),
//        "challenge": ("filelist_places365-challenge.tar", "70a8307e459c3de41690a7c76c931734"),
//    }
// # (file, md5)
//        _CATEGORIES_META = ("categories_places365.txt", "06c963b85866bd0649f97cb43dd16673")
//        # {split: (file, md5)}
//        _FILE_LIST_META = {
//        "train-standard": ("places365_train_standard.txt", "30f37515461640559006b8329efbed1a"),
//        "train-challenge": ("places365_train_challenge.txt", "b2931dc997b8c33c27e7329c073a6b57"),
//        "val": ("places365_val.txt", "e9f2fd57bfd9d07630173f4e8708e4b1"),
//    }
// # {(split, small): (file, md5)}
//        _IMAGES_META = {
//        ("train-standard", False): ("train_large_places365standard.tar", "67e186b496a84c929568076ed01a8aa1"),
//        ("train-challenge", False): ("train_large_places365challenge.tar", "605f18e68e510c82b958664ea134545f"),
//        ("val", False): ("val_large.tar", "9b71c4993ad89d2d8bcbdc4aef38042f"),
//        ("train-standard", True): ("train_256_places365standard.tar", "53ca1c756c3d1e7809517cc47c5561c5"),
//        ("train-challenge", True): ("train_256_places365challenge.tar", "741915038a5e3471ec7332404dfb64ef"),
//        ("val", True): ("val_256.tar", "e27b17d8d44f4af9a78502beb927f808"),
//    }



       void load_data();

       void check_resources();

   };

}
