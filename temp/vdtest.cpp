#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <torch/torch.h>
#include "../include/media/opencv/videos.h"


int main() {
//    std::vector<cv::Mat> frames;
    fs::path videoPath("/home/kami/Documents/temp/videos/people-detection.mp4");
    vector<torch::Tensor> frames = torch::ext::media::opencv::videos::extractVideoFramesAsTensor(videoPath);
    cout << frames.size() << endl;
    cout << frames[0] << endl;

    return 0;
}