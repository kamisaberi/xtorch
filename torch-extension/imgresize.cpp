#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Image not found or cannot be read." << std::endl;
        return 1;
    }

    int new_width = 320;
    int new_height = 240;
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

    cv::imwrite("output.jpg", resized_image);
    return 0;
}