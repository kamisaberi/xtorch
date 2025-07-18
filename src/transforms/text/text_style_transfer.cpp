#include <transforms/text/text_style_transfer.h>

#include <stdexcept>

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>
#include <map>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};


// --- A Mock Style Transfer Client for the Example ---
// This class simulates a powerful AI model. In a real application, the 'transfer'
// method would call a neural network.
class MockStyleTransferClient : public xt::transforms::text::StyleTransferClient {
public:
    auto transfer(const std::string& text, const std::string& target_style) const -> std::string override {
        if (target_style == "formal") {
            if (text == "Hey, we gotta fix this bug ASAP.") {
                return "Dear team, it is imperative that we resolve this issue promptly.";
            }
        } else if (target_style == "informal") {
            if (text == "The data indicates a significant statistical anomaly.") {
                return "Whoa, the numbers look super weird.";
            }
        } else if (target_style == "shakespearean") {
            if (text == "I'm very hungry and want to get a burger.") {
                return "Hark, a great famine doth besiege me; I must acquire a beef-ed patty forthwith!";
            }
        }
        // Default fallback if no rule matches
        return "Style transfer for '" + text + "' to style '" + target_style + "' is not supported by this mock.";
    }
};


int main() {
    // 1. --- Setup ---
    // Instantiate our mock client. This could be a real client in a real app.
    auto model_client = std::make_shared<MockStyleTransferClient>();

    // 2. --- Create Multiple Transformers from the Same Client ---
    // We create a specific transform instance for each style we want to generate.
    xt::transforms::text::TextStyleTransfer to_formal(model_client, "formal");
    xt::transforms::text::TextStyleTransfer to_informal(model_client, "informal");
    xt::transforms::text::TextStyleTransfer to_shakespearean(model_client, "shakespearean");

    // 3. --- Run the Transformations ---
    std::string informal_sentence = "Hey, we gotta fix this bug ASAP.";
    std::string formal_sentence = "The data indicates a significant statistical anomaly.";
    std::string modern_sentence = "I'm very hungry and want to get a burger.";

    std::cout << "--- Applying 'formal' style transfer ---" << std::endl;
    std::cout << "Original:  \"" << informal_sentence << "\"" << std::endl;
    auto formal_result = std::any_cast<std::string>(to_formal.forward({informal_sentence}));
    std::cout << "Formal:    \"" << formal_result << "\"\n" << std::endl;

    std::cout << "--- Applying 'informal' style transfer ---" << std::endl;
    std::cout << "Original:  \"" << formal_sentence << "\"" << std::endl;
    auto informal_result = std::any_cast<std::string>(to_informal.forward({formal_sentence}));
    std::cout << "Informal:  \"" << informal_result << "\"\n" << std::endl;

    std::cout << "--- Applying 'shakespearean' style transfer ---" << std::endl;
    std::cout << "Original:     \"" << modern_sentence << "\"" << std::endl;
    auto shakespearean_result = std::any_cast<std::string>(to_shakespearean.forward({modern_sentence}));
    std::cout << "Shakespearean:\"" << shakespearean_result << "\"" << std::endl;

    return 0;
}
*/

namespace xt::transforms::text {

    TextStyleTransfer::TextStyleTransfer(std::shared_ptr<StyleTransferClient> client, const std::string& target_style)
            : client_(client), target_style_(target_style) {

        if (!client_) {
            throw std::invalid_argument("StyleTransferClient provided to TextStyleTransfer must not be null.");
        }
        if (target_style_.empty()) {
            throw std::invalid_argument("Target style cannot be an empty string.");
        }
    }

    auto TextStyleTransfer::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("TextStyleTransfer::forward received an empty list.");
        }

        std::string input_text;
        try {
            input_text = std::any_cast<std::string>(any_vec[0]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Input to TextStyleTransfer must be of type std::string.");
        }

        if (input_text.empty()) {
            return std::string(""); // Return empty string if input is empty
        }

        // 2. --- Core Logic ---
        // Delegate the complex task of style transfer to the client.
        // The transform itself just orchestrates the call with its configured style.
        std::string transferred_text = client_->transfer(input_text, target_style_);

        // 3. --- Return Result ---
        return transferred_text;
    }

} // namespace xt::transforms::text