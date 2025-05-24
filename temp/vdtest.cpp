#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <torch/torch.h>
#include "include/utils/videos.h"

using namespace std;

int main() {
//    std::vector<cv::Mat> frames;
    std::filesystem::path videoPath("/home/kami/Documents/temp/videos/people-detection.mp4");
    vector<torch::Tensor> frames = xt::utils::videos::extractVideoFramesAsTensor(videoPath);
    cout << frames.size() << endl;
    cout << frames[0] << endl;

    return 0;
}