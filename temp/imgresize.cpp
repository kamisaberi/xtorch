#include <opencv2/opencv.hpp>
#include "../include/media/opencv/images.h"
#include <iostream>
#include <torch/torch.h>
#include <filesystem>
namespace fs = std::filesystem;

int main() {

  fs::path pth = fs::path("/home/kami/Documents/temp/food-101/images/apple_pie/134.jpg");

  torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(pth , {320,240} );
  cout << tensor.sizes() << endl;

//
//
//    cv::Mat image = cv::imread("/home/kami/Documents/temp/food-101/images/apple_pie/134.jpg", cv::IMREAD_COLOR);
//    if (image.empty()) {
//        std::cerr << "Error: Image not found or cannot be read." << std::endl;
//        return 1;
//    }
//
//    int new_width = 320;
//    int new_height = 240;
//    std::cout << image.rows << " " << image.cols << std::endl;
////    cv::Mat resized_image;
//    cv::resize(image, image, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
//    std::cout << image.rows << " " << image.cols << std::endl;
////    cv::imwrite("output.jpg", image);
    return 0;
}