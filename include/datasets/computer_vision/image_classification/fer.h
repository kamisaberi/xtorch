#pragma once

#include "datasets/common.h"


using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {
    class FER2013 : public xt::datasets::Dataset {
        /*
            `FER2013
            <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge>`_ Dataset.

            .. note::
                This dataset can return test labels only if ``fer2013.csv`` OR
                ``icml_face_data.csv`` are present in ``root/fer2013/``. If only
                ``train.csv`` and ``test.csv`` are present, the test labels are set to
                ``None``.

            Args:
                root (str or ``pathlib.Path``): Root directory of dataset where directory
                    ``root/fer2013`` exists. This directory may contain either
                    ``fer2013.csv``, ``icml_face_data.csv``, or both ``train.csv`` and
                    ``test.csv``. Precendence is given in that order, i.e. if
                    ``fer2013.csv`` is present then the rest of the files will be
                    ignored. All these (combinations of) files contain the same data and
                    are supported for convenience, but only ``fer2013.csv`` and
                    ``icml_face_data.csv`` are able to return non-None test labels.
                split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
                transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
                    version. E.g, ``transforms.RandomCrop``
                target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        */
    public :

        explicit FER2013(const std::string& root);
        FER2013(const std::string& root, xt::datasets::DataMode mode);
        FER2013(const std::string& root, xt::datasets::DataMode mode, bool download);
        FER2013(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        FER2013(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);




    private :

        // resources = {
        //     "train": ("train.csv", "3f0dfb3d3fd99c811a1299cb947e3131"),
        //     "test": ("test.csv", "b02c2298636a634e8c2faabbf3ea9a23"),
        //     "fer": ("fer2013.csv", "f8428a1edbd21e88f42c73edd2a14f95"),
        //     "icml": ("icml_face_data.csv", "b114b9e04e6949e5fe8b6a98b3892b1d"),
        // }

        //TODO fs::path dataset_folder_name = "?";
        fs::path dataset_folder_name = "?";
        bool download = false;
        fs::path root;
        fs::path dataset_path;


        void load_data();

        void check_resources();


    };
}
