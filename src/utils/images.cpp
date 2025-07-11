#include "../../include/utils/images.h"


namespace xt::utils::image {
    torch::Tensor convertImageToTensor(std::filesystem::path img, vector<int> size) {
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

    cv::Mat tensor_to_mat_local(torch::Tensor tensor) {
        tensor = tensor.squeeze().detach().clone();
        tensor = tensor.permute({1, 2, 0}); // CHW -> HWC

        // IMPORTANT: We assume the tensor is float and needs to be handled as such.
        // OpenCV can work with float Mats directly (CV_32FCn).
        cv::Mat mat(tensor.size(0), tensor.size(1), CV_32FC(tensor.size(2)));
        std::memcpy(mat.data, tensor.data_ptr<float>(), sizeof(float) * tensor.numel());
        return mat;
    }

    // Helper function to convert a cv::Mat back to a tensor.
    torch::Tensor mat_to_tensor_local(const cv::Mat& mat) {
        // Ensure mat is float type as expected
        cv::Mat mat_float;
        if (mat.type() != CV_32FC3 && mat.type() != CV_32FC1) {
            mat.convertTo(mat_float, CV_32F);
        } else {
            mat_float = mat;
        }

        cv::Mat mat_cont = mat_float.isContinuous() ? mat_float : mat_float.clone();

        torch::Tensor tensor = torch::from_blob(mat_cont.data, {mat_cont.rows, mat_cont.cols, mat_cont.channels()}, torch::kFloat32);
        tensor = tensor.permute({2, 0, 1}); // HWC -> CHW
        return tensor.clone(); // Clone to make sure the tensor owns its memory
    }

    cv::Mat tensor_to_mat_8u(torch::Tensor tensor) {
        tensor = tensor.squeeze().detach().clone(); // Remove channel/batch dims

        // Convert float tensor [0, 1] to uint8 tensor [0, 255]
        tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);

        cv::Mat mat(tensor.size(0), tensor.size(1), CV_8UC1);
        std::memcpy(mat.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());
        return mat;
    }

}
