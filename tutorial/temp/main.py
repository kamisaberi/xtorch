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
                {"Text Classification": "examples/natural_language_processing/3_1_text_classification.markdown"},
                {"Sequence to Sequence": "examples/natural_language_processing/3_2_sequence_to_sequence.markdown"},
                {"Language Modeling": "examples/natural_language_processing/3_3_language_modeling.markdown"},
            ]},
            {"Audio and Speech": [
                {"Speech Recognition": "examples/audio_and_speech/4_1_speech_recognition.markdown"},
                {"Audio Classification": "examples/audio_and_speech/4_2_audio_classification.markdown"},
            ]},
            {"Time Series and Sequential Data": [
                {"Forecasting": "examples/time_series_and_sequential_data/5_1_forecasting.markdown"},
                {"Anomaly Detection": "examples/time_series_and_sequential_data/5_2_anomaly_detection.markdown"},
            ]},
            {"Reinforcement Learning": [
                {"Value-Based Methods": "examples/reinforcement_learning/6_1_value_based_methods.markdown"},
                {"Policy-Based Methods": "examples/reinforcement_learning/6_2_policy_based_methods.markdown"},
            ]},
            {"Graph Neural Networks": [
                {"Node-Level Tasks": "examples/graph_neural_networks/7_1_node_level_tasks.markdown"},
                {"Graph-Level Tasks": "examples/graph_neural_networks/7_2_graph_level_tasks.markdown"},
            ]},
            {"Generative Models": [
                {"Autoencoders": "examples/generative_models/8_1_autoencoders.markdown"},
                {"GANs": "examples/generative_models/8_2_gans.markdown"},
                {"Diffusion Models": "examples/generative_models/8_3_diffusion_models.markdown"},
            ]},
            {"Deployment and Production": [
                {"Model Serialization": "examples/deployment_and_production/9_1_model_serialization.markdown"},
                {"Inference": "examples/deployment_and_production/9_2_inference.markdown"},
                {"Web Services": "examples/deployment_and_production/9_3_web_services.markdown"},
            ]},
            {"Data Handling and Preprocessing": [
                {"Datasets": "examples/data_handling_and_preprocessing/10_1_datasets.markdown"},
                {"Data Loaders": "examples/data_handling_and_preprocessing/10_2_data_loaders.markdown"},
                {"Transforms": "examples/data_handling_and_preprocessing/10_3_transforms.markdown"},
            ]},
            {"Optimization and Training Techniques": [
                {"Optimizers": "examples/optimization_and_training_techniques/11_1_optimizers.markdown"},
                {"Learning Rate Schedulers": "examples/optimization_and_training_techniques/11_2_learning_rate_schedulers.markdown"},
                {"Regularization": "examples/optimization_and_training_techniques/11_3_regularization.markdown"},
            ]},
            {"Performance and Benchmarking": [
                {"Speed Optimization": "examples/performance_and_benchmarking/12_1_speed_optimization.markdown"},
                {"Memory Management": "examples/performance_and_benchmarking/12_2_memory_management.markdown"},
            ]},
            {"Distributed and Parallel Training": [
                {"Data Parallelism": "examples/distributed_and_parallel_training/13_1_data_parallelism.markdown"},
                {"Model Parallelism": "examples/distributed_and_parallel_training/13_2_model_parallelism.markdown"},
                {"Distributed Training": "examples/distributed_and_parallel_training/13_3_distributed_training.markdown"},
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
