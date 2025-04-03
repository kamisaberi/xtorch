#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
   class SEMEION : BaseDataset {
       /*
       r"""`SEMEION <http://archive.ics.uci.edu/ml/datasets/semeion+handwritten+digit>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``semeion.py`` exists.
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
       SEMEION(const std::string &root);
       SEMEION(const std::string &root, DataMode mode);
       SEMEION(const std::string &root, DataMode mode , bool download);
       SEMEION(const std::string &root, DataMode mode , bool download, TransformType transforms);


   private:
       fs::path url = fs::path("http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data");
       fs::path dataset_file_name = fs::path("semeion.data");
       std::string dataset_file_md5 = "cb545d371d2ce14ec121470795a77432";
       fs::path dataset_folder_name = "semeion";

       void load_data();

       void check_resources();

    };
}
