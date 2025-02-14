#include <torch/torch.h>
//#include <torch/data/datasets/mnist.h>
//#include <vector>
//#include <fstream>
#include <iostream>
//#include <string>
//#include <filesystem>
//#include <curl/curl.h>
#include "include/datasets/cifar.h"

int main() {
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;
    std::cout << "Hello World" << std::endl;
    std::string r = "./test/test1/ali";
//    torch::ext::data::datasets::UCF101 u1 = torch::ext::data::datasets::UCF101(5);
//    torch::ext::data::datasets::UCF101 u1 = torch::ext::data::datasets::UCF101(r);

    torch::ext::data::datasets::CIFAR100 cifar100("/home/kami/Documents/temp/", true , true);
    cout << cifar100.get(0).target << endl;
    std::cout << "End\n";


    return 0;
}

