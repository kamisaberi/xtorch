#pragma once
#include "../headers/datasets.h"
#include "base.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class RenderedSST2 : BaseDataset {
        /*
        """`The Rendered SST2 Dataset <https://github.com/openai/CLIP/blob/main/data/rendered-sst2.md>`_.

    Rendered SST2 is an image classification dataset used to evaluate the models capability on optical
    character recognition. This dataset was generated by rendering sentences in the Standford Sentiment
    Treebank v2 dataset.

    This dataset contains two classes (positive and negative) and is divided in three splits: a  train
    split containing 6920 images (3610 positive and 3310 negative), a validation split containing 872 images
    (444 positive and 428 negative), and a test split containing 1821 images (909 positive and 912 negative).

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), `"val"` and ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

         */
    public :
        RenderedSST2(const std::string &root);
        RenderedSST2(const std::string &root, DataMode mode);
        RenderedSST2(const std::string &root, DataMode mode , bool download);
        RenderedSST2(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private:
        fs::path url = fs::path("https://openaipublic.azureedge.net/clip/data/rendered-sst2.tgz");
        fs::path dataset_file_name = "rendered-sst2.tgz";
        std::string dataset_file_md5 = "2384d08e9dcfa4bd55b324e610496ee5";
        fs::path dataset_folder_name = "rendered-sst2";
        void load_data();

        void check_resources();

    };
}
