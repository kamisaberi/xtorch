// #include <transforms/signal/mfcc.h>
//
//
// // /*
// // // Example Usage (goes in a main.cpp or test file)
// // #include "include/utils/audio/io.h"
// // #include <iostream>
// //
// // int main() {
// //     // 1. Load an audio file.
// //     auto [waveform, sample_rate] = xt::utils::audio::load("some_speech_audio.wav");
// //
// //     // 2. Create the MFCC transform. Let's ask for 13 MFCCs from 40 Mel bins.
// //     xt::transforms::signal::MFCC mfcc_transform(
// //         sample_rate,
// //         /*n_mfcc=*/13,
// //         /*n_fft=*/512,
// //         /*win_length=*/0,
// //         /*hop_length=*/160,
// //         /*f_min=*/0.0,
// //         /*f_max=*/c10::nullopt,
// //         /*n_mels=*/40
// //     );
// //
// //     // 3. Apply the transform to get the MFCC features.
// //     torch::Tensor mfccs = std::any_cast<torch::Tensor>(mfcc_transform.forward({waveform}));
// //
// //     // 4. Verify the output shape.
// //     // Shape should be (n_mfcc, num_frames), e.g., (13, ...)
// //     std::cout << "Resulting MFCC shape: " << mfccs.sizes() << std::endl;
// //
// //     // These MFCCs are now ready features for a speech recognition model.
// //
// //     return 0;
// // }
// // */
//
// namespace xt::transforms::signal {
//
//     // Computes the Type-II DCT matrix.
//     torch::Tensor MFCC::create_dct_matrix(int n_mels, int n_mfcc) {
//         auto options = torch::TensorOptions().dtype(torch::kFloat32);
//         torch::Tensor dct_matrix = torch::empty({n_mfcc, n_mels}, options);
//
//         // Create vectors for n (input bins) and k (output coefficients)
//         auto n = torch::arange(0, n_mels, options);
//         auto k = torch::arange(0, n_mfcc, options).unsqueeze(1); // shape (n_mfcc, 1)
//
//         // Compute the cosine term using broadcasting
//         dct_matrix = torch::cos(M_PI / n_mels * (n + 0.5) * k);
//
//         // Apply scaling factors
//         // First row (k=0) is scaled by sqrt(1/N)
//         // Other rows (k>0) are scaled by sqrt(2/N)
//         auto scales = torch::full({n_mfcc}, std::sqrt(2.0 / n_mels), options);
//         scales[0] = std::sqrt(1.0 / n_mels);
//
//         dct_matrix *= scales.unsqueeze(1); // shape (n_mfcc, 1)
//
//         return dct_matrix;
//     }
//
//     MFCC::MFCC(
//         int sample_rate,
//         int n_mfcc,
//         int n_fft,
//         int win_length,
//         int hop_length,
//         double f_min,
//         c10::optional<double> f_max,
//         int n_mels,
//         bool log_mels)
//         : mel_spectrogram_( // Initialize the composed MelSpectrogram transform
//             sample_rate, n_fft, (win_length > 0 ? win_length : n_fft), hop_length,
//             f_min, f_max, n_mels, /*power=*/2.0),
//           log_mels_(log_mels) {
//
//         if (n_mfcc > n_mels) {
//             throw std::invalid_argument("n_mfcc cannot be greater than n_mels.");
//         }
//         // Pre-compute the DCT matrix for efficiency.
//         dct_matrix_ = create_dct_matrix(n_mels, n_mfcc);
//     }
//
//     auto MFCC::forward(std::initializer_list<std::any> tensors) -> std::any {
//         // --- 1. Input Validation ---
//         std::vector<std::any> any_vec(tensors);
//         if (any_vec.empty()) {
//             throw std::invalid_argument("MFCC::forward received an empty list.");
//         }
//         // The input is a raw waveform.
//         torch::Tensor waveform = std::any_cast<torch::Tensor>(any_vec[0]);
//
//         // --- 2. Compute Mel Spectrogram ---
//         // Call the forward pass of the internal MelSpectrogram object.
//         torch::Tensor mel_spec = std::any_cast<torch::Tensor>(
//             mel_spectrogram_.forward({waveform})
//         );
//
//         // --- 3. Apply Logarithm ---
//         if (log_mels_) {
//             mel_spec = torch::log(mel_spec + log_offset_);
//         }
//
//         // --- 4. Apply DCT ---
//         // Matrix multiplication with the pre-computed DCT matrix.
//         // Shapes: (n_mfcc, n_mels) @ (..., n_mels, time) -> (..., n_mfcc, time)
//         torch::Tensor mfccs = torch::matmul(dct_matrix_.to(mel_spec.device()), mel_spec);
//
//
//
//
//
//         return mfccs;
//     }
//
// } // namespace xt::transforms::signal