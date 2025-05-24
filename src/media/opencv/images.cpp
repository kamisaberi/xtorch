#include "../../../include/media/opencv/images.h"


namespace xt::utils::image {
    torch::Tensor convertImageToTensor(fs::path img, vector<int> size) {
        cv::Mat image = cv::imread(img.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Could not load image at: " + img.string());
        }
        // 2. Convert BGR (OpenCV default) to RGB
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        // 3. Convert image data to float and normalize to [0, 1]
        // image.convertTo(image, CV_32F, 1.0 / 255.0);
        image.convertTo(image, CV_32F);
        if (size[0] != 0 && size[1] != 0) {
            cv::resize(image, image, cv::Size(size[0], size[1]), 0, 0, cv::INTER_LINEAR);
        }
        // image.convertTo(image, CV_32F, 1.0 / 255.0);
        // cout << "BE:" << image.rows << "  " << image.cols << "  " << image.channels() << endl;

        // cout << "AF:" << image.rows << "  " << image.cols << "  " << image.channels() << endl;

        // 4. Create a tensor from the image data
        // OpenCV uses HWC (Height, Width, Channels) format
        torch::Tensor tensor = torch::from_blob(image.data, {image.rows, image.cols, image.channels()},
                                                torch::kFloat);


        // 5. Permute to CHW (Channels, Height, Width) format, which is PyTorch's default
        tensor = tensor.permute({2, 0, 1});

        // 6. Make sure the tensor is contiguous in memory
        tensor = tensor.contiguous();
        return tensor;
    }


    torch::Tensor resize(const torch::Tensor &tensor, const std::vector<int64_t> &size) {
        return torch::nn::functional::interpolate(
                tensor.unsqueeze(0),
                torch::nn::functional::InterpolateFuncOptions().size(size).mode(
                        torch::kBilinear).align_corners(false)
        ).squeeze(0);
    }

}
