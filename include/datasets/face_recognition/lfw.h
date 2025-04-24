#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {
    class LFW : public BaseDataset {
    public :
        explicit LFW(const std::string &root);
        LFW(const std::string &root, DataMode mode);
        LFW(const std::string &root, DataMode mode , bool download);
        LFW(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        fs::path dataset_folder_name = fs::path("lfw-py");
        fs::path url = fs::path("http://vis-www.cs.umass.edu/lfw/");
        std::map<std::string, std::tuple<std::string, fs::path, std::string> > file_dict = {
            {"original", {"lfw", fs::path("lfw.tgz"), "a17d05bd522c52d84eca14327a23d494"}},
            {"funneled", {"lfw_funneled", fs::path("lfw-funneled.tgz"), "1b42dfed7d15c9b2dd63d5e5840c86ad"}},
            {
                "deepfunneled",
                {"lfw-deepfunneled", fs::path("lfw-deepfunneled.tgz"), "68331da3eb755a505a502b5aacb3c201"}
            }
        };

        std::map<std::string, std::string> checksums = {
            {"pairs.txt", "9f1ba174e4e1c508ff7cdf10ac338a7d"},
            {"pairsDevTest.txt", "5132f7440eb68cf58910c8a45a2ac10b"},
            {"pairsDevTrain.txt", "4f27cbf15b2da4a85c1907eb4181ad21"},
            {"people.txt", "450f0863dd89e85e73936a6d71a3474b"},
            {"peopleDevTest.txt", "e4bf5be0a43b5dcd9dc5ccfcb8fb19c5"},
            {"peopleDevTrain.txt", "54eaac34beb6d042ed3a7d883e247a21"},
            {"lfw-names.txt", "a6d0a479bd074669f656265a6e693f6d"}
        };

        std::map<std::string, std::string> annot_file = {
            {"10fold", ""},
            {"train", "DevTrain"},
            {"test", "DevTest"}
        };
        fs::path names = fs::path("lfw-names.txt");

        void load_data();

        void check_resources();
    };

    class LFWPeople : public LFW {
        /*
        """`LFW <http://vis-www.cs.umass.edu/lfw/>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``lfw-py`` exists or will be saved to if download is set to True.
        split (string, optional): The image split to use. Can be one of ``train``, ``test``,
            ``10fold`` (default).
        image_set (str, optional): Type of image funneling to use, ``original``, ``funneled`` or
            ``deepfunneled``. Defaults to ``funneled``.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomRotation``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

         */
    public :
        explicit LFWPeople(const std::string &root);
        LFWPeople(const std::string &root, DataMode mode);
        LFWPeople(const std::string &root, DataMode mode , bool download);
        LFWPeople(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };

    class LFWPairs : public LFW {
        /*
        """`LFW <http://vis-www.cs.umass.edu/lfw/>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``lfw-py`` exists or will be saved to if download is set to True.
        split (string, optional): The image split to use. Can be one of ``train``, ``test``,
            ``10fold``. Defaults to ``10fold``.
        image_set (str, optional): Type of image funneling to use, ``original``, ``funneled`` or
            ``deepfunneled``. Defaults to ``funneled``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomRotation``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """


         */
    public :
        explicit  LFWPairs(const std::string &root);
        LFWPairs(const std::string &root, DataMode mode);
        LFWPairs(const std::string &root, DataMode mode , bool download);
        LFWPairs(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };
}
