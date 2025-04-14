#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include "../include/datasets/specific/mnist.h"
#include "../include/models/cnn/lenet5.h"
#include <torch/data/transforms/base.h>
#include <functional>
#include "../include/definitions/transforms.h"

#include "../include/datasets/specific/mnist.h"
#include "../include/datasets/image-classification/cifar.h"
#include "../include/datasets/specific/imagenette.h"
#include "../include/models/cnn/lenet5.h"
#include "../include/definitions/transforms.h"
#include "../include/data-loaders/data-loader.h"
#include "../include/trainers/trainer.h"
#include <type_traits>
#include <iostream>
#include "../include/datasets/general/csv-dataset.h"


int main() {
    auto dataset = xt::data::datasets::CSVDataset("/home/kami/Documents/temp/iris.csv");



    return 0;
}
