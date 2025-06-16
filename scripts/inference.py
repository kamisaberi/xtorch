# inference.py
# A script to test the final, self-contained model file.

import torch
import os

# --- Configuration ---
# The path to the final model created by the C++/Python pipeline.
MODEL_PATH = "independent_model.pt"

# The input shape must match what the model was traced with in converter.py.
# In our case, it was [batch_size, input_features] = [1, 1].
INPUT_SHAPE = [1, 1]

print("--- Final Model Inference Test ---")

# 1. Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file not found at '{MODEL_PATH}'")
    print("Please run the './trainer' executable first to generate the model.")
    exit(1)

# 2. Load the TorchScript model
#    This is the key step. `torch.jit.load` reads the self-contained file
#    and reconstructs the model graph and its trained weights.
try:
    print(f"Loading model from '{MODEL_PATH}'...")
    model = torch.jit.load(MODEL_PATH)
    # It's always good practice to set the model to evaluation mode for inference.
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading the model: {e}")
    exit(1)

# You can uncomment the line below to see the model's architecture graph
# that was saved inside the file.
# print(model.graph)

# 3. Create a dummy input tensor for testing
#    The C++ trainer learned the function y = 5x - 2.
#    Let's test with an input of 10. The output should be close to (5 * 10 - 2) = 48.
test_value = 10.0
input_tensor = torch.tensor([[test_value]], dtype=torch.float32)

print(f"\nCreated a test input tensor with value: {test_value}")

# 4. Run inference
#    Since this is a traced model, we don't need to track gradients.
with torch.no_grad():
    try:
        # Pass the input tensor to the loaded model.
        output = model(input_tensor)
        print("✅ Inference executed successfully.")
    except Exception as e:
        print(f"❌ An error occurred during model execution: {e}")
        exit(1)

# 5. Print the results
print("\n--- Test Results ---")
print(f"Input: {input_tensor.item():.2f}")
print(f"Predicted Output: {output.item():.4f}")
print(f"(Expected value is approximately 48.0)")

if abs(output.item() - 48.0) < 5.0: # Allow for some training inaccuracy
    print("\n✅ Verification PASSED: The output is close to the expected value.")
else:
    print("\n⚠️ Verification WARNING: The output is not close to the expected value.")