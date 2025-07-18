#include <transforms/text/back_translation.h>


// Example Main - Uncomment to run a standalone test
// #include <iostream>
//
// // Mock xt::Module for testing purposes
// class xt::Module {
// public:
//     virtual ~Module() = default;
//     virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
// };
//
// // --- Mock Translation Clients for Demonstration ---
//
// // A mock client that "translates" from English to German
// class MockEnglishToGermanClient : public xt::transforms::text::TranslationClient {
// public:
//     auto translate(const std::string& text) -> std::string override {
//         if (text == "The quick brown fox jumps over the lazy dog.") {
//             return "Der schnelle braune Fuchs springt über den faulen Hund.";
//         }
//         return "Unbekannter Text"; // Unknown text
//     }
// };
//
// // A mock client that "translates" from German back to English, but paraphrased.
// class MockGermanToEnglishClient : public xt::transforms::text::TranslationClient {
// public:
//     auto translate(const std::string& text) -> std::string override {
//         if (text == "Der schnelle braune Fuchs springt über den faulen Hund.") {
//             // This is the paraphrased result
//             return "A fast brown fox leaps over the lazy dog.";
//         }
//         return "Unknown text";
//     }
// };
//
// int main() {
//     // 1. --- Setup ---
//     // In a real application, these clients would connect to actual translation models.
//     // Here, we use our mock clients.
//     auto to_german_client = std::make_shared<MockEnglishToGermanClient>();
//     auto to_english_client = std::make_shared<MockGermanToEnglishClient>();
//
//     // Create the transform by injecting the clients.
//     xt::transforms::text::BackTranslation back_translator(to_german_client, to_english_client);
//
//     // 2. --- Run the Transform ---
//     std::string original_text = "The quick brown fox jumps over the lazy dog.";
//     std::cout << "Original Text:    \"" << original_text << "\"" << std::endl;
//
//     std::string augmented_text = std::any_cast<std::string>(back_translator.forward({original_text}));
//
//     std::cout << "Augmented Text:   \"" << augmented_text << "\"" << std::endl;
//
//     // 3. --- Verify that the output is different ---
//     if (original_text != augmented_text) {
//         std::cout << "\nSuccessfully created a paraphrased sentence!" << std::endl;
//     }
//
//     return 0;
// }

namespace xt::transforms::text {

    BackTranslation::BackTranslation(
            std::shared_ptr<TranslationClient> to_intermediate_client,
            std::shared_ptr<TranslationClient> from_intermediate_client
    ) : to_intermediate_client_(to_intermediate_client),
        from_intermediate_client_(from_intermediate_client) {

        if (!to_intermediate_client_ || !from_intermediate_client_) {
            throw std::invalid_argument("Translation clients provided to BackTranslation must not be null.");
        }
    }

    auto BackTranslation::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("BackTranslation::forward received an empty list.");
        }

        std::string input_text;
        try {
            input_text = std::any_cast<std::string>(any_vec[0]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Input to BackTranslation must be of type std::string.");
        }

        if (input_text.empty()) {
            return std::string(""); // Return empty string if input is empty
        }

        // 2. --- Translate to Intermediate Language ---
        std::string intermediate_text = to_intermediate_client_->translate(input_text);

        // 3. --- Translate Back to Source Language ---
        std::string back_translated_text = from_intermediate_client_->translate(intermediate_text);

        // 4. --- Return Result ---
        return back_translated_text;
    }

} // namespace xt::transforms::text