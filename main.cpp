//#include <torch/torch.h>
//#include <torch/data/datasets/mnist.h>
//#include <vector>
//#include <fstream>
//#include <iostream>
//#include <string>
//#include <filesystem>
//#include <curl/curl.h>
#include "include/datasets/cifar100.h"


int main() {

    torch::data::datasets::CIFAR100 cifar100= torch::data::datasets::CIFAR100("/home/kami/Documents/temp/", true , true);


    return 0;
}

