#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {
    class MovingMNIST : xt::datasets::Dataset {
        /*
        """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where ``MovingMNIST/mnist_test_seq.npy`` exists.
        split (string, optional): The dataset split, supports ``None`` (default), ``"train"`` and ``"test"``.
            If ``split=None``, the full data is returned.
        split_ratio (int, optional): The split ratio of number of frames. If ``split="train"``, the first split
            frames ``data[:, :split_ratio]`` is returned. If ``split="test"``, the last split frames ``data[:, split_ratio:]``
            is returned. If ``split=None``, this parameter is ignored and the all frames data is returned.
        transform (callable, optional): A function/transform that takes in a torch Tensor
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

         */
    public :
        explicit MovingMNIST(const std::string& root);
        MovingMNIST(const std::string& root, xt::datasets::DataMode mode);
        MovingMNIST(const std::string& root, xt::datasets::DataMode mode, bool download);
        MovingMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        MovingMNIST(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);


    private :
        fs::path url = fs::path("http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy");
        fs::path dataset_file_name = fs::path("mnist_test_seq.npy");
        string dataset_file_md5 = "be083ec986bfe91a449d63653c411eb2";

        // TODO fs::path dataset_folder_name
        fs::path dataset_folder_name = "?";

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void load_data();

        void check_resources();

    };
}
