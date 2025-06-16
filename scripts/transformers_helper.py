import torch
from transformers import AutoModelForImageClassification
import sys

if len(sys.argv) != 2:
    print("Usage: python convert_model.py <output_path>")
    sys.exit(1)

output_path = sys.argv[1]
model_name = "microsoft/resnet-18"

try:
    print(f"[Python] Loading model '{model_name}' from Hugging Face...")
    hf_model = AutoModelForImageClassification.from_pretrained(model_name)
    hf_model.eval()

    # THIS IS THE CRITICAL WRAPPER
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            outputs = self.model(x)
            return outputs.logits

    model_to_trace = ModelWrapper(hf_model)

    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"[Python] Tracing model with input shape: {dummy_input.shape}")

    traced_model = torch.jit.trace(model_to_trace, dummy_input)
    traced_model.save(output_path)

    print(f"\n[Python] Success! Model has been converted to TorchScript and saved at '{output_path}'.")
    sys.exit(0)

except Exception as e:
    print(f"[Python] Error: {e}", file=sys.stderr)
    sys.exit(1)