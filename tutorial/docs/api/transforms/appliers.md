# Transform Appliers

Appliers, or meta-transforms, are a special category of transformations that do not modify the data directly. Instead, they control *how* and *when* other transforms are applied.

They are the building blocks for creating complex, dynamic, and probabilistic data augmentation pipelines. By combining simple transforms with appliers, you can construct sophisticated augmentation strategies.

All applier transforms are located under the `xt::transforms` namespace and can be found in the `<xtorch/transforms/appliers/>` header directory.

---

## `Compose`

This is the most fundamental and widely used applier. It chains a list of transforms together and applies them sequentially. It is the primary tool for building any multi-step data processing pipeline.

`Compose` is covered in detail on the main [Transforms page](index.md), but its usage is recapped here for completeness.

### Usage
```cpp
auto pipeline = std::make_unique<xt::transforms::Compose>(
    std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{256, 256}),
    std::make_shared<xt::transforms::image::CenterCrop>(std::vector<int64_t>{224, 224}),
    std::make_shared<xt::transforms::general::Normalize>(mean, std)
);
```

---

## `RandomApply`

Applies a given transform with a specific probability. This is essential for data augmentation, as you often don't want to apply a transformation to every single sample.

### Usage
```cpp
// This pipeline will apply Gaussian Blur to 30% of the images that pass through it.
auto pipeline = std::make_unique<xt::transforms::Compose>(
    std::make_shared<xt::transforms::RandomApply>(
        std::make_shared<xt::transforms::image::GaussianBlur>(/*kernel_size=*/3),
        /*p=*/0.3 // The probability of application
    )
);
```

---

## `OneOf`

Takes a list of transforms and randomly selects and applies **exactly one** of them to the data. This is useful when you want to choose from a set of mutually exclusive augmentations.

### Usage
```cpp
// This pipeline will apply EITHER Gaussian Blur OR Motion Blur to each image.
auto pipeline = std::make_unique<xt::transforms::Compose>(
    std::make_shared<xt::transforms::OneOf>(
        std::vector<std::shared_ptr<xt::Module>>{
            std::make_shared<xt::transforms::image::GaussianBlur>(3),
            std::make_shared<xt::transforms::image::MotionBlur>(/*kernel_size=*/5)
        }
    )
);
```

---

## `SomeOf`

Takes a list of transforms and an integer `n`, then randomly selects and applies **exactly `n`** of them to the data.

### Usage
```cpp
// This pipeline will randomly select and apply exactly TWO of the three listed transforms.
auto pipeline = std::make_unique<xt::transforms::Compose>(
    std::make_shared<xt::transforms::SomeOf>(
        std::vector<std::shared_ptr<xt::Module>>{
            std::make_shared<xt::transforms::image::GaussianBlur>(3),
            std::make_shared<xt::transforms::image::Sharpen>(),
            std::make_shared<xt::transforms::image::Emboss>()
        },
        /*n=*/2 // Apply 2 transforms from the list
    )
);
```

---

## `Sometimes`

Applies a list of transforms sequentially, but each transform is only applied with a given probability `p`. This is like a `Compose` where every step is wrapped in `RandomApply`.

### Usage
```cpp
// This pipeline will go through three steps.
// First, it has a 50% chance of applying a flip.
// Then, it has a 30% chance of applying a rotation.
// Finally, it has a 25% chance of applying noise.
auto pipeline = std::make_unique<xt::transforms::Compose>(
    std::make_shared<xt::transforms::Sometimes>(
        std::vector<std::shared_ptr<xt::Module>>{
            std::make_shared<xt::transforms::image::RandomHorizontalFlip>(),
            std::make_shared<xt::transforms::image::RandomRotation>(15),
            std::make_shared<xt::transforms::image::GaussianNoise>()
        },
        /*p=*/0.5 // Default probability if not specified per-transform
    )
);
```

---

## Other Appliers

| Applier | Description | Header File |
|---|---|---|
| `Repeat` | Applies a given transform `n` times in a row. | `appliers/repeat.h` |
| `ReplayCompose` | A variant of `Compose` designed for deterministic augmentations on paired data (e.g., applying the exact same random crop to both an image and its segmentation mask). | `appliers/replay_compose.h` |
| `Palindrome` | Applies a sequence of transforms and then applies them again in reverse order. | `appliers/palindrome.h` |
