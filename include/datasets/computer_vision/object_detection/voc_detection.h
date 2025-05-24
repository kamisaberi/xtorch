#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets
{
    class VOCDetection : public xt::datasets::Dataset
    {
        /*
        """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

     Args:
         root (str or ``pathlib.Path``): Root directory of the VOC Dataset.
         year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
         image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
             ``year=="2007"``, can also be ``"test"``.
         download (bool, optional): If true, downloads the dataset from the internet and
             puts it in root directory. If dataset is already downloaded, it is not
             downloaded again.
             (default: alphabetic indexing of VOC's 20 classes).
         transform (callable, optional): A function/transform that takes in a PIL image
             and returns a transformed version. E.g, ``transforms.RandomCrop``
         target_transform (callable, required): A function/transform that takes in the
             target and transforms it.
         transforms (callable, optional): A function/transform that takes input sample and its target as entry
             and returns a transformed version.
     """

         */

    public :
        explicit VOCDetection(const std::string& root);
        VOCDetection(const std::string& root, xt::datasets::DataMode mode);
        VOCDetection(const std::string& root, xt::datasets::DataMode mode, bool download);
        VOCDetection(const std::string& root, xt::datasets::DataMode mode, bool download,
                     std::unique_ptr<xt::Module> transformer);
        VOCDetection(const std::string& root, xt::datasets::DataMode mode, bool download,
                     std::unique_ptr<xt::Module> transformer,
                     std::unique_ptr<xt::Module> target_transformer);

    private :
        // TODO fs::path dataset_folder_name
        fs::path dataset_folder_name = "?";

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void load_data();

        void check_resources();
    };
}
