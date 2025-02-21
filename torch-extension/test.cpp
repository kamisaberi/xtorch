#include <torch/torch.h>
//#include <torch/data/datasets/mnist.h>
//#include <vector>
//#include <fstream>
#include <iostream>
//#include <string>
//#include <filesystem>
//#include <curl/curl.h>
#include "include/datasets/mnist.h"

int main() {
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;
    std::cout << "Hello World" << std::endl;
    std::string r = "./test/test1/ali";
//    torch::ext::data::datasets::UCF101 u1 = torch::ext::data::datasets::UCF101(5);
//    torch::ext::data::datasets::UCF101 u1 = torch::ext::data::datasets::UCF101(r);

    torch::ext::data::datasets::MNIST train("/home/kami/Documents/temp/", true , true);
    torch::ext::data::datasets::MNIST test("/home/kami/Documents/temp/", false , true);
//    cout << mnist.get(0).target << endl;
    std::cout << "End\n";


    return 0;
}

