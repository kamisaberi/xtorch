#include <transforms/video/frame_interpolation.h>

#include <stdexcept>

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};


// --- A Mock Frame Interpolation Client for the Example ---
// This class simulates a deep learning model by performing simple linear
// interpolation (a cross-fade effect). This is useful for testing the pipeline
// as it produces tensors of the correct shape and a visually intuitive result.
class MockFrameInterpolatorClient : public xt::transforms::video::FrameInterpolatorClient {
public:
    auto interpolate(
        const torch::Tensor& start_frame,
        const torch::Tensor& end_frame,
        int num_to_generate
    ) const -> torch::Tensor override {

        std::vector<torch::Tensor> new_frames;
        new_frames.reserve(num_to_generate);

        for (int i = 1; i <= num_to_generate; ++i) {
            // Calculate the interpolation weight (alpha)
            float alpha = static_cast<float>(i) / (num_to_generate + 1);

            // new_frame = (1 - alpha) * start + alpha * end
            torch::Tensor interpolated_frame = torch::lerp(start_frame, end_frame, alpha);
            new_frames.push_back(interpolated_frame);
        }

        // Stack the list of frame tensors into a single tensor of shape (N, C, H, W)
        return torch::stack(new_frames, 0);
    }
};


int main() {
    // 1. --- Setup ---
    // Instantiate our mock client. In a real application, this would be a client
    // that loads and runs a real TorchScript or ONNX model.
    auto model_client = std::make_shared<MockFrameInterpolatorClient>();

    // Create a transform to insert 3 new frames between each pair.
    int frames_to_insert = 3;
    xt::transforms::video::FrameInterpolation frame_interpolator(model_client, frames_to_insert);

    // 2. --- Create Dummy Frame Data ---
    // Let's create two simple 3x4x4 frames (C, H, W).
    // start_frame will be all zeros (black).
    // end_frame will be all ones (white).
    torch::Tensor start_frame = torch::zeros({3, 4, 4});
    torch::Tensor end_frame = torch::ones({3, 4, 4});

    std::cout << "Input frames shape: " << start_frame.sizes() << std::endl;
    std::cout << "Requesting to insert " << frames_to_insert << " new frames." << std::endl;

    // 3. --- Run the Transform ---
    auto result_any = frame_interpolator.forward({start_frame, end_frame});

    // 4. --- Verify the Output ---
    try {
        auto generated_frames = std::any_cast<torch::Tensor>(result_any);

        std::cout << "\nOutput tensor shape: " << generated_frames.sizes() << std::endl;

        // The shape should be (num_to_insert, C, H, W) -> (3, 3, 4, 4)
        if (generated_frames.size(0) == frames_to_insert) {
            std::cout << "Successfully generated the correct number of frames." << std::endl;
        }

        // You can also inspect the values to see the fade effect.
        // The first generated frame should have values around 0.25.
        // The second generated frame should have values around 0.50.
        // The third generated frame should have values around 0.75.
        std::cout << "Value of the first pixel in the first generated frame:\n" << generated_frames[0][0][0][0] << std::endl;
        std::cout << "Value of the first pixel in the second generated frame:\n" << generated_frames[1][0][0][0] << std::endl;
        std::cout << "Value of the first pixel in the third generated frame:\n" << generated_frames[2][0][0][0] << std::endl;


    } catch(const std::bad_any_cast& e) {
        std::cerr << "Failed to cast result to torch::Tensor." << std::endl;
    }

    return 0;
}
*/


namespace xt::transforms::video {

    FrameInterpolation::FrameInterpolation(std::shared_ptr<FrameInterpolatorClient> client, int num_to_insert)
        : client_(client), num_to_insert_(num_to_insert) {

        if (!client_) {
            throw std::invalid_argument("FrameInterpolatorClient provided to FrameInterpolation must not be null.");
        }
        if (num_to_insert_ <= 0) {
            throw std::invalid_argument("Number of frames to insert must be positive.");
        }
    }

    auto FrameInterpolation::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("FrameInterpolation::forward requires exactly two tensors (start_frame, end_frame).");
        }

        torch::Tensor start_frame, end_frame;
        try {
            start_frame = std::any_cast<torch::Tensor>(any_vec[0]);
            end_frame = std::any_cast<torch::Tensor>(any_vec[1]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Inputs to FrameInterpolation must be of type torch::Tensor.");
        }

        if (!start_frame.defined() || !end_frame.defined()) {
            throw std::invalid_argument("Input tensors passed to FrameInterpolation are not defined.");
        }
        if (start_frame.sizes() != end_frame.sizes()) {
            throw std::invalid_argument("Start and end frames must have the same shape.");
        }
        if (start_frame.dim() != 3) {
             throw std::invalid_argument("Input frames must be 3-dimensional tensors (C, H, W).");
        }

        // 2. --- Core Logic ---
        // Delegate the complex task of interpolation to the client.
        torch::Tensor generated_frames = client_->interpolate(start_frame, end_frame, num_to_insert_);

        // 3. --- Return Result ---
        return generated_frames;
    }

} // namespace xt::transforms::video