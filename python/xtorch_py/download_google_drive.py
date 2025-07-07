import sys
import argparse
import os
import gdown

def main():
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face image classification model to TorchScript.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Full path where the TorchScript model (.pt file) will be saved."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/resnet-18",
        help="Name of the Hugging Face model from the Hub (e.g., 'google/vit-base-patch16-224')."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the dummy input used during tracing."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,  # Allow inferring from model config if possible
        help="Height and width for the dummy input image (e.g., 224 for 224x224). "
             "If not provided, attempts to infer from model config or feature extractor."
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Number of image channels for the dummy input (e.g., 3 for RGB)."
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force model loading and tracing on CPU, even if CUDA is available."
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbose type (0: NONE, 1: ERRORS,2: EVERYTHING) "
    )

    args = parser.parse_args()

    # Determine device
    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verbose = args.verbose
    if verbose == 2:
        print(f"[Python Script] Using device: {device}")
        print(f"[Python Script] --- Configuration ---")
        print(f"[Python Script] Hugging Face Model: {args.model_name}")
        print(f"[Python Script] Output TorchScript Path: {args.output_path}")
        print(f"[Python Script] Dummy Input Batch Size: {args.batch_size}")
        print(f"[Python Script] Dummy Input Channels: {args.channels}")
        print(f"[Python Script] Verbose: {args.verbose}")

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            if verbose == 2:
                print(f"[Python Script] Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        # Load model from Hugging Face
        if verbose == 2:
            print(f"[Python Script] Loading model '{args.model_name}' from Hugging Face Hub...")
        # For image classification, AutoModelForImageClassification is generally suitable.
        # If the model has a specific config for image size, it might be used.
        hf_model = AutoModelForImageClassification.from_pretrained(args.model_name).to(device)
        hf_model.eval()  # Set to evaluation mode (important for dropout, batchnorm layers)

        # Determine image size for dummy input
        image_size_to_use = args.image_size
        if image_size_to_use is None:
            if verbose == 2:
                print("[Python Script] Attempting to infer image size from model config or feature extractor...")
            try:
                # Try to get from model's config (common for ViT, ConvNeXT)
                config = AutoConfig.from_pretrained(args.model_name)
                if hasattr(config, 'image_size') and isinstance(config.image_size, int):
                    image_size_to_use = config.image_size
                    if verbose == 2:
                        print(f"[Python Script] Inferred image_size from model config: {image_size_to_use}")
                elif hasattr(config, 'input_size') and isinstance(config.input_size, (list, tuple)) and len(
                        config.input_size) >= 2:  # e.g. timm models
                    image_size_to_use = config.input_size[-1]  # Often H, W or C, H, W
                    if verbose == 2:
                        print(f"[Python Script] Inferred image_size from model config.input_size: {image_size_to_use}")
            except Exception as e_config:
                if verbose == 2:
                    print(f"[Python Script] Could not get image_size from model config: {e_config}")

            if image_size_to_use is None:  # If still None, try feature extractor
                try:
                    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
                    if hasattr(feature_extractor, 'size'):  # Common attribute
                        # 'size' can be an int or a dict like {'shortest_edge': 224} or {'height': 224, 'width': 224}
                        if isinstance(feature_extractor.size, int):
                            image_size_to_use = feature_extractor.size
                        elif isinstance(feature_extractor.size, dict):
                            # Prioritize height/width if available, otherwise shortest_edge
                            if 'height' in feature_extractor.size and 'width' in feature_extractor.size:
                                if feature_extractor.size['height'] == feature_extractor.size['width']:
                                    image_size_to_use = feature_extractor.size['height']
                                else:  # Non-square, pick one (e.g., height, or make it an error/warning)
                                    image_size_to_use = feature_extractor.size['height']
                                    if verbose == 2:
                                        print(
                                            f"[Python Script] Warning: Inferred non-square size from feature_extractor, using height: {image_size_to_use}")
                            elif 'shortest_edge' in feature_extractor.size:
                                image_size_to_use = feature_extractor.size['shortest_edge']
                        if verbose == 2:
                            print(f"[Python Script] Inferred image_size from feature extractor: {image_size_to_use}")
                except Exception as e_feat:
                    if verbose == 1 or verbose == 2:
                        print(f"[Python Script] Could not get image_size from feature extractor: {e_feat}")

            if image_size_to_use is None:  # If still none after all attempts
                image_size_to_use = 224  # Fallback to a common default
                if verbose == 2:
                    print(f"[Python Script] Could not infer image_size. Defaulting to: {image_size_to_use}")

        if verbose == 2:
            print(f"[Python Script] Using Dummy Input Image Size (HxW): {image_size_to_use}x{image_size_to_use}")

        # Wrap the model
        model_to_trace = ModelWrapper(hf_model).to(device)
        model_to_trace.eval()

        # Create dummy input tensor
        dummy_input = torch.randn(
            args.batch_size,
            args.channels,
            image_size_to_use,
            image_size_to_use
        ).to(device)

        if verbose == 2:
            print(
                f"[Python Script] Tracing model with dummy input shape: {list(dummy_input.shape)} on device: {dummy_input.device}")

        # Trace the model to convert to TorchScript
        # Using check_trace=False can sometimes help with complex models, but it's good to keep it True if possible.
        traced_model = torch.jit.trace(model_to_trace, dummy_input, strict=False)  # strict=False can be more lenient

        # Save the TorchScript model
        traced_model.save(args.output_path)

        if verbose == 2:
            print(f"\n[Python Script] SUCCESS!")
            print(f"[Python Script] Model '{args.model_name}' converted to TorchScript.")
            print(f"[Python Script] Saved at: '{args.output_path}'")
        sys.exit(0)  # Exit with success code

    except Exception as e:
        if verbose == 1 or verbose == 2:
            print(f"\n[Python Script] ERROR: An exception occurred during model conversion.", file=sys.stderr)
            print(f"[Python Script] Model: {args.model_name}", file=sys.stderr)
            print(f"[Python Script] Exception: {type(e).__name__}: {e}", file=sys.stderr)
        # For more detailed debugging, uncomment the next line:
        # import traceback
        # print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    main()
