#include "../../include/utils/images.h"


namespace xt::utils::image
{
    torch::Tensor convertImageToTensor(std::filesystem::path img, vector<int> size)
    {
        cv::Mat image = cv::imread(img.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw std::runtime_error("Could not load image at: " + img.string());
        }
        // 2. Convert BGR (OpenCV default) to RGB
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        // 3. Convert image data to float and normalize to [0, 1]
        // image.convertTo(image, CV_32F, 1.0 / 255.0);
        image.convertTo(image, CV_32F);
        if (size[0] != 0 && size[1] != 0)
        {
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


    torch::Tensor resize(const torch::Tensor& tensor, const std::vector<int64_t>& size)
    {
        return torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions().size(size).mode(
                torch::kBilinear).align_corners(false)
        ).squeeze(0);
    }

    cv::Mat tensor_to_mat_local(torch::Tensor tensor)
    {
        tensor = tensor.squeeze().detach().clone();
        tensor = tensor.permute({1, 2, 0}); // CHW -> HWC

        // IMPORTANT: We assume the tensor is float and needs to be handled as such.
        // OpenCV can work with float Mats directly (CV_32FCn).
        cv::Mat mat(tensor.size(0), tensor.size(1), CV_32FC(tensor.size(2)));
        std::memcpy(mat.data, tensor.data_ptr<float>(), sizeof(float) * tensor.numel());
        return mat;
    }

    // Helper function to convert a cv::Mat back to a tensor.
    torch::Tensor mat_to_tensor_local(const cv::Mat& mat)
    {
        // Ensure mat is float type as expected
        cv::Mat mat_float;
        if (mat.type() != CV_32FC3 && mat.type() != CV_32FC1)
        {
            mat.convertTo(mat_float, CV_32F);
        }
        else
        {
            mat_float = mat;
        }

        cv::Mat mat_cont = mat_float.isContinuous() ? mat_float : mat_float.clone();

        torch::Tensor tensor = torch::from_blob(mat_cont.data, {mat_cont.rows, mat_cont.cols, mat_cont.channels()},
                                                torch::kFloat32);
        tensor = tensor.permute({2, 0, 1}); // HWC -> CHW
        return tensor.clone(); // Clone to make sure the tensor owns its memory
    }

    cv::Mat tensor_to_mat_8u(torch::Tensor tensor)
    {
        tensor = tensor.squeeze().detach().clone(); // Remove channel/batch dims

        // Convert float tensor [0, 1] to uint8 tensor [0, 255]
        tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);

        cv::Mat mat(tensor.size(0), tensor.size(1), CV_8UC1);
        std::memcpy(mat.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());
        return mat;
    }


    torch::Tensor mat_to_tensor_float(const cv::Mat& mat)
    {
        if (mat.empty())
        {
            throw std::invalid_argument("Input cv::Mat to mat_to_tensor_float is empty.");
        }

        // Ensure the Mat is continuous in memory for from_blob.
        // If not, clone it to make it continuous.
        cv::Mat mat_cont = mat.isContinuous() ? mat : mat.clone();

        // Determine the shape for the tensor.
        // Note: OpenCV Mats are HWC (Height, Width, Channels).
        std::vector<int64_t> shape = {mat_cont.rows, mat_cont.cols};
        if (mat_cont.channels() > 1)
        {
            shape.push_back(mat_cont.channels());
        }

        // Create a tensor from the blob of data.
        // This does NOT copy the data and does not take ownership.
        torch::Tensor tensor = torch::from_blob(mat_cont.data, shape, torch::kByte);

        // Convert the 8-bit integer tensor (0-255) to a 32-bit float tensor.
        tensor = tensor.to(torch::kFloat32);

        // Normalize the float tensor to the [0.0, 1.0] range.
        tensor = tensor.div(255.0f);

        // Permute the dimensions from HWC to the standard CHW format for deep learning.
        if (tensor.dim() == 3)
        {
            // For color images
            tensor = tensor.permute({2, 0, 1});
        }
        else
        {
            // For grayscale images, add a channel dimension to make it [1, H, W]
            tensor = tensor.unsqueeze(0);
        }

        // It is good practice to clone the tensor at the end.
        // This ensures the new tensor owns its memory and is not just a view
        // into the memory of the (potentially temporary) cv::Mat.
        return tensor.clone();
    }
}
