#include "include/transforms/video/optical_flow_warping.h"

#include <stdexcept>

// These would typically be in a separate utils file, but are included here for a self-contained example.
// namespace xt { namespace utils { namespace image {
//     // Basic tensor (C,H,W, float) to mat (H,W,C, float) conversion
//     cv::Mat tensor_to_mat_float(const torch::Tensor& tensor) {
//         torch::Tensor permuted = tensor.permute({1, 2, 0}).contiguous();
//         cv::Mat mat(permuted.size(0), permuted.size(1), CV_32FC(permuted.size(2)), permuted.data_ptr<float>());
//         return mat.clone();
//     }
//     // Basic mat (H,W,C, float) to tensor (C,H,W, float) conversion
//     torch::Tensor mat_to_tensor_float(const cv::Mat& mat) {
//         torch::Tensor tensor = torch::from_blob(mat.data, {mat.rows, mat.cols, mat.channels()}, torch::kFloat32);
//         return tensor.permute({2, 0, 1}).clone();
//     }
// }}}


// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};


// --- A Mock Optical Flow Client for the Example ---
// This client doesn't compute real optical flow. Instead, it generates a
// constant, uniform flow field where every pixel moves 15 pixels to the right
// and 10 pixels down. This is perfect for verifying that the warping logic works.
class MockOpticalFlowClient : public xt::transforms::video::OpticalFlowClient {
public:
    auto calculate_flow(const cv::Mat& frame1, const cv::Mat& frame2) const -> cv::Mat override {
        // Create a 2-channel float matrix (CV_32FC2) of the same HxW as the input frames.
        cv::Mat flow_field(frame1.size(), CV_32FC2);

        // Define the uniform motion vector (dx, dy).
        cv::Vec2f motion_vector(15.0f, 10.0f); // 15px right, 10px down

        // Fill the entire flow field with this constant vector.
        flow_field = motion_vector;

        return flow_field;
    }
};


int main() {
    // 1. --- Setup ---
    auto model_client = std::make_shared<MockOpticalFlowClient>();
    xt::transforms::video::OpticalFlowWarping warper(model_client);

    // 2. --- Create Dummy Frame Data ---
    // Create a 100x100 source frame with a small 10x10 white square at position (20, 30).
    torch::Tensor source_frame = torch::zeros({1, 100, 100}, torch::kFloat32);
    source_frame.slice(1, 30, 40).slice(2, 20, 30).fill_(1.0f); // H slice, then W slice

    // The target frame can be anything, as our mock client ignores it.
    torch::Tensor target_frame = torch::zeros({1, 100, 100}, torch::kFloat32);

    std::cout << "Created a source frame with a white square at (x=20, y=30)." << std::endl;
    std::cout << "Mock flow is a uniform shift of (dx=15, dy=10)." << std::endl;
    std::cout << "Expected new square position is (x=35, y=40)." << std::endl;

    // 3. --- Run the Transform ---
    auto result_any = warper.forward({source_frame, target_frame});

    // 4. --- Verify the Output ---
    try {
        auto warped_frame = std::any_cast<torch::Tensor>(result_any);
        std::cout << "\nOutput tensor shape: " << warped_frame.sizes() << std::endl;

        // Check the pixel value at the original top-left corner of the square. It should be 0 now.
        float old_pos_val = warped_frame.slice(1, 30, 31).slice(2, 20, 21).item<float>();

        // Check the pixel value at the NEW, expected top-left corner. It should be 1.
        float new_pos_val = warped_frame.slice(1, 40, 41).slice(2, 35, 36).item<float>();

        std::cout << "Pixel value at original pos (20,30): " << old_pos_val << std::endl;
        std::cout << "Pixel value at new pos (35,40):      " << new_pos_val << std::endl;

        if (old_pos_val < 0.1f && new_pos_val > 0.9f) {
            std::cout << "\nVerification successful! The square was correctly warped." << std::endl;
        }

    } catch(const std::bad_any_cast& e) {
        std::cerr << "Failed to cast result to torch::Tensor." << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::video {

    OpticalFlowWarping::OpticalFlowWarping(std::shared_ptr<OpticalFlowClient> client)
        : client_(client) {
        if (!client_) {
            throw std::invalid_argument("OpticalFlowClient provided to OpticalFlowWarping must not be null.");
        }
    }

    auto OpticalFlowWarping::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("OpticalFlowWarping::forward requires two tensors (source_frame, target_frame).");
        }

        torch::Tensor source_tensor, target_tensor;
        try {
            source_tensor = std::any_cast<torch::Tensor>(any_vec[0]);
            target_tensor = std::any_cast<torch::Tensor>(any_vec[1]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Inputs to OpticalFlowWarping must be of type torch::Tensor.");
        }

        // 2. --- Data Conversion (Tensor -> Mat) ---
        // Convert tensors to OpenCV Mats. Optical flow is often computed on grayscale images.
        cv::Mat source_mat_color = xt::utils::image::tensor_to_mat_float(source_tensor);
        cv::Mat target_mat_color = xt::utils::image::tensor_to_mat_float(target_tensor);

        cv::Mat source_gray, target_gray;
        cv::cvtColor(source_mat_color, source_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(target_mat_color, target_gray, cv::COLOR_BGR2GRAY);

        // Convert grayscale to 8-bit, which is standard input for many flow algorithms
        source_gray.convertTo(source_gray, CV_8U, 255.0);
        target_gray.convertTo(target_gray, CV_8U, 255.0);

        // 3. --- Calculate Optical Flow ---
        cv::Mat flow = client_->calculate_flow(source_gray, target_gray);

        // 4. --- Perform Warping using cv::remap ---
        // cv::remap needs two maps (map_x, map_y) that specify, for each pixel in the
        // destination image, which source coordinate to get the color from.
        cv::Mat map_x(flow.size(), CV_32FC1);
        cv::Mat map_y(flow.size(), CV_32FC1);

        for (int y = 0; y < flow.rows; ++y) {
            for (int x = 0; x < flow.cols; ++x) {
                cv::Vec2f flow_at_point = flow.at<cv::Vec2f>(y, x);
                // For destination pixel (x,y), we look up the source pixel (x + dx, y + dy)
                map_x.at<float>(y, x) = x + flow_at_point[0];
                map_y.at<float>(y, x) = y + flow_at_point[1];
            }
        }

        // Apply the remap
        cv::Mat warped_mat;
        cv::remap(source_mat_color, warped_mat, map_x, map_y, cv::INTER_LINEAR);

        // 5. --- Data Conversion (Mat -> Tensor) & Return ---
        return xt::utils::image::mat_to_tensor_float(warped_mat);
    }

} // namespace xt::transforms::video