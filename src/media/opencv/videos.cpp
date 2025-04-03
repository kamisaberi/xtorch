#include "../../../include/media/opencv/videos.h"



namespace torch::ext::media::opencv::videos {


    std::vector<cv::Mat> extractFrames(const std::string& videoFilePath ) {
        std::vector<cv::Mat> frames;
        cv::VideoCapture cap(videoFilePath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file." << std::endl;
            return frames;
        }

        cv::Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) {
                break;
            }
            frames.push_back(frame);
        }
        cap.release();
        return frames;
    }


    std::vector<torch::Tensor> extractVideoFramesAsTensor(fs::path videoFilePath) {
        std::vector<torch::Tensor> frames;
        cv::VideoCapture cap(videoFilePath.string());
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file." << std::endl;
            return frames;
        }

        cv::Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) {
                break;
            }

            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            frame.convertTo(frame, CV_32F);
            torch::Tensor tensor = torch::from_blob(frame.data, {frame.rows, frame.cols, frame.channels()},
                                                    torch::kFloat32
            );
            tensor = tensor.permute({2, 0, 1});
            tensor = tensor.contiguous();
            frames.push_back(tensor);
        }
        cap.release();
        return frames;
    }



}
