# converter.py
import torch
import torch.nn as nn
import struct
from collections import OrderedDict

# 1. Define the IDENTICAL model architecture in Python.
#    This must exactly match the C++ 'struct Net'.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- HELPER FUNCTION TO LOAD THE RAW WEIGHTS ---
def load_parameters(file_path):
    state_dict = OrderedDict()
    with open(file_path, "rb") as f:
        # Read the number of parameters
        num_params, = struct.unpack("I", f.read(4))

        for _ in range(num_params):
            # Read parameter name
            name_len, = struct.unpack("I", f.read(4))
            param_name = f.read(name_len).decode("utf-8")

            # Read tensor shape
            dims, = struct.unpack("I", f.read(4))
            shape = struct.unpack(f"{dims}q", f.read(dims * 8))

            # Read tensor data
            num_elements = torch.prod(torch.tensor(shape)).item()
            num_bytes = num_elements * 4 # Assuming float32

            raw_data = f.read(num_bytes)
            tensor = torch.frombuffer(raw_data, dtype=torch.float32).view(shape)

            state_dict[param_name] = tensor
            print(f"Loaded '{param_name}' with shape {list(shape)}")

    return state_dict

print("--- Step 2: Python Converter ---")

# 2. Create an instance of the Python model.
py_model = Net()
py_model.eval()

# 3. Load the weights from the C++ binary file.
WEIGHTS_FILE = "cpp_trained_weights.bin"
try:
    print(f"Loading raw weights from '{WEIGHTS_FILE}'...")
    loaded_state_dict = load_parameters(WEIGHTS_FILE)
    py_model.load_state_dict(loaded_state_dict)
    print("\n✅ Success: Weights loaded into Python model.")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    exit()

# 4. Create the self-contained, independent model.
#    This is the crucial step. We trace the Python model that now
#    holds the weights trained by C++.
try:
    print("\nTracing the model to create an independent file...")
    dummy_input = torch.randn(1, 1) # Input shape must match model's input
    traced_model = torch.jit.trace(py_model, dummy_input)

    # 5. Save the final model.
    INDEPENDENT_MODEL_PATH = "independent_model.pt"
    traced_model.save(INDEPENDENT_MODEL_PATH)
    print(f"✅✅✅ FINAL SUCCESS: Independent model saved to '{INDEPENDENT_MODEL_PATH}'")
    print("This file can be loaded in C++ without needing the model definition.")

except Exception as e:
    print(f"❌ Error during tracing or saving: {e}")
    exit()