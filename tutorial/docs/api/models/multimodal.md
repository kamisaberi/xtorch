# Multimodal Models

Multimodal models are a sophisticated class of neural networks designed to process and relate information from two or more different data types (modalities), such as images, text, and audio. These models can learn rich, joint representations that capture the relationships between different modalities.

A key application is connecting images with text, enabling tasks like text-to-image retrieval, image captioning, and zero-shot image classification.

xTorch provides implementations of several powerful multimodal architectures. All multimodal models are located under the `xt::models` namespace and their headers can be found in the `<xtorch/models/multimodal/>` directory.

## General Usage

The usage of multimodal models is highly specific to the architecture, as they require multiple, distinct inputs. The `forward` method is often replaced by more descriptive methods like `encode_image` and `encode_text`.

For instance, a model like CLIP (Contrastive Language-Image Pre-training) has two main components:
1.  **An Image Encoder**: Typically a Vision Transformer (ViT) that processes an image and outputs a single feature vector (embedding).
2.  **A Text Encoder**: Typically a Transformer that processes a sequence of text tokens and outputs a single feature vector.

These models are trained to map related images and text descriptions to nearby points in a shared embedding space.

```cpp
#include <xtorch/xtorch.hh>
#include <iostream>

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // --- Instantiate a CLIP Model ---
    // The constructor might take configuration for the vision and text models.
    xt::models::CLIP model(/*vision_config*/, /*text_config*/);
    model.to(device);
    model.eval(); // Pre-trained models are often used in evaluation mode

    std::cout << "CLIP Model Instantiated." << std::endl;

    // --- Create Dummy Input Data ---
    // A batch of preprocessed images
    auto images = torch::randn({4, 3, 224, 224}).to(device);
    // A batch of tokenized and numericalized text
    auto text = torch::randint(0, 49408, {4, 77}).to(device); // Vocab size, sequence length

    // --- Perform Encoding ---
    // Note: The actual method names might differ. Check the header file.
    auto image_features = model.encode_image(images);
    auto text_features = model.encode_text(text);

    // The features are normalized to have a unit norm
    image_features /= image_features.norm(2, -1, true);
    text_features /= text_features.norm(2, -1, true);

    std::cout << "Image features shape: " << image_features.sizes() << std::endl;
    std::cout << "Text features shape: " << text_features.sizes() << std::endl;

    // --- Calculate Similarity ---
    // The dot product between the image and text features gives the cosine similarity.
    // This can be used to find the best text description for each image.
    auto similarity_logits = torch::matmul(image_features, text_features.t()) * model.logit_scale.exp();

    std::cout << "Similarity matrix shape: " << similarity_logits.sizes() << std::endl;
}
```

---

## Available Multimodal Models

xTorch provides the following multimodal architectures:

| Model | Description | Header File |
|---|---|---|
| `CLIP` | **Contrastive Language-Image Pre-training**. Learns a joint embedding space for images and text, enabling powerful zero-shot classification and text-to-image retrieval. | `clip.h` |
| `ViLBERT`| **Vision-and-Language BERT**. A model for learning task-agnostic joint representations of image content and natural language. | `vilbert.h` |
| `BLIP` | **Bootstrapping Language-Image Pre-training**. A model for unified vision-language understanding and generation that introduces a novel bootstrapping method. | `blip.h` |
| `Flamingo` | A family of Visual Language Models (VLM) that can perform few-shot learning on a variety of vision-language tasks. | `flamingo.h` |
| `LLaVA` | **Large Language and Vision Assistant**. A model that connects a vision encoder with a large language model to enable visual instruction following. | `llava.h` |
| `MERT` | A self-supervised representation learning model for music understanding that can process both audio and symbolic (e.g., MIDI) music data. | `mert.h` |
