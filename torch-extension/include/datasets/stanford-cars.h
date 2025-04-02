#pragma once
#include "../headers/datasets.h"
#include "base.h"

namespace xt::data::datasets {
   class StanfordCars : BaseDataset {
/*
       """Stanford Cars  Dataset

           The Cars dataset contains 16,185 images of 196 classes of cars. The data is
           split into 8,144 training images and 8,041 testing images, where each class
           has been split roughly in a 50-50 split

           The original URL is https://ai.stanford.edu/~jkrause/cars/car_dataset.html, but it is broken.
           Follow the instructions in ``download`` argument to obtain and use the dataset offline.

           .. note::

               This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

           Args:
               root (str or ``pathlib.Path``): Root directory of dataset
               split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
               transform (callable, optional): A function/transform that takes in a PIL image
                   and returns a transformed version. E.g, ``transforms.RandomCrop``
               target_transform (callable, optional): A function/transform that takes in the
                   target and transforms it.
               download (bool, optional): This parameter exists for backward compatibility but it does not
                   download the dataset, since the original URL is not available anymore. The dataset
                   seems to be available on Kaggle so you can try to manually download and configure it using
                   `these instructions <https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616>`_,
                   or use an integrated
                   `dataset on Kaggle <https://github.com/pytorch/vision/issues/7545#issuecomment-2282674373>`_.
                   In both cases, first download and configure the dataset locally, and use the dataset with
                   ``"download=False"``.
           """

 */
   public :
       StanfordCars(const std::string &root);
       StanfordCars(const std::string &root, DataMode mode);
       StanfordCars(const std::string &root, DataMode mode , bool download);
       StanfordCars(const std::string &root, DataMode mode , bool download, TransformType transforms);

   private :
       void load_data(DataMode mode = DataMode::TRAIN);

       void check_resources(const std::string &root, bool download = false);

   };
}
