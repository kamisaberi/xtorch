

# File videos.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**media**](dir_aa03a1d12037901d4378cbd73498762d.md) **>** [**opencv**](dir_2b794fa5f0369c1c80752771b4d33858.md) **>** [**videos.h**](videos_8h.md)

[Go to the documentation of this file](videos_8h.md)


```C++
#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <torch/torch.h>
#include <string>
#include <vector>

using namespace std;

namespace fs = std::filesystem;

namespace torch::ext::media::opencv::videos {
    std::vector<cv::Mat>  extractFrames(const std::string& videoFilePath);
    std::vector<torch::Tensor> extractVideoFramesAsTensor(fs::path videoFilePath);
}
```


