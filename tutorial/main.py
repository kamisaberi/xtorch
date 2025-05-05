import os

BUILD_REF = False


def define_env(env):
    """
    Define the MkDocs navigation dynamically based on environment variables.
    """
    is_rtd = os.getenv("READTHEDOCS") == "True"

    nav = [
        {"Home": "index.md"},
        {"Getting Started": [
            {"Installation": "getting_started/installation.md"},
            {"Quickstart": "getting_started/quickstart_tutorial.md"},
            {"Migration from PyTorch": "getting_started/migration_guide.md"},
        ]},
        {"Tutorials": [
            {"Basics": [
                {"Hello World": "tutorials/basics/hello_world.md"},
                {"Tensors": "tutorials/basics/working_with_tensors.md"},
                {"Custom Layers": "tutorials/basics/custom_layers.md"},
            ]},
            {"Intermediate": [
                {"GPU Acceleration": "tutorials/intermediate/gpu_acceleration.md"},
                {"Mixed Precision": "tutorials/intermediate/mixed_precision.md"},
            ]},
            {"Advanced": [
                {"JIT Compilation": "tutorials/advanced/jit_compilation.md"},
                {"C++ Extensions": "tutorials/advanced/cpp_extensions.md"},
            ]}
        ]},
        {"Examples": [
            {"Getting Started": "examples/getting_started/1_getting_started.markdown"},
            {"Computer Vision": [
                {"Image Classification": "examples/computer_vision/2_1_image_classification.markdown"},
                {"Object Detection": "examples/computer_vision/2_2_object_detection.markdown"},
                {"Segmentation": "examples/computer_vision/2_3_segmentation.markdown"},
                {"Image Generation": "examples/computer_vision/2_4_image_generation.markdown"},
            ]},
            {"Natural Language Processing": [
                {"Text Classification": "examples/natural_language_processing/2_1_image_classification.markdown"},
                {"Sequence to Sequence": "examples/natural_language_processing/2_2_object_detection.markdown"},
                {"Language Modeling": "examples/natural_language_processing/2_3_segmentation.markdown"},
            ]},
            {"Audio and Speech": [
                {"Speech Recognition": "examples/natural_language_processing/2_1_image_classification.markdown"},
                {"Audio Classification": "examples/natural_language_processing/2_2_object_detection.markdown"},
            ]},
            {"Time Series and Sequential Data": [
                {"Forecasting": "examples/natural_language_processing/2_1_image_classification.markdown"},
                {"Anomaly Detection": "examples/natural_language_processing/2_2_object_detection.markdown"},
            ]},
            {"Reinforcement Learning": [
                {"Value-Based Methods": "examples/natural_language_processing/2_1_image_classification.markdown"},
                {"Policy-Based Methods": "examples/natural_language_processing/2_2_object_detection.markdown"},
            ]},
            {"Graph Neural Networks": [
                {"Node-Level Tasks": "examples/natural_language_processing/2_1_image_classification.markdown"},
                {"Graph-Level Tasks": "examples/natural_language_processing/2_2_object_detection.markdown"},
            ]},
            {"Examples": "examples/0_examples.markdown"},
        ]},
        {"Roadmap": [
            {"Datasets": "roadmaps/datasets.md"},
            {"Models": "roadmaps/models.md"},
            {"Transforms": "roadmaps/transforms.md"},
        ]},
        {"Ecosystem": [
            {"XTorch Vision": "ecosystem/vision.md"},
            {"XTorch NLP": "ecosystem/nlp.md"},
            {"Integration Guide": "ecosystem/integration.md"},
        ]},
        {"Developer Resources": [
            {"Contributing": "developer/contributing.md"},
            {"Testing": "developer/testing.md"},
            {"Benchmarking": "developer/benchmarks.md"},
        ]},
        {"Community": [
            {"FAQ": "community/faq.md"},
            {"Code of Conduct": "community/code_of_conduct.md"},
        ]},
    ]
    if BUILD_REF:
        if not is_rtd:
            api_ref = {"API Reference": []}
            # Add API Reference entries
            # api_ref = nav[2]["API Reference"]
            api_ref["API Reference"].append({"Index": "api/index.md"})

            api_ref["API Reference"].extend([
                {"Python API": [
                    {"Core": "api/python/core.md"},
                    {"NN Modules": "api/python/nn.md"},
                    {"Optimizers": "api/python/optim.md"},
                    {"Utilities": "api/python/utils.md"},
                ]},
                {"C++ API": [
                    {"Tensor Operations": "api/cpp/tensor.md"},
                    {"Core Classes": "api/cpp/core_classes.md"},
                    {"Native Extensions": "api/cpp/native.md"},
                ]}
            ])
            # nav.append(api_ref)
        else:
            api_ref = {"API Reference": "api/ref.md"}

        nav.append(api_ref)
    # Set the navigation
    env.conf["nav"] = nav
