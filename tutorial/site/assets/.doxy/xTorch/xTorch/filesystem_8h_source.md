

# File filesystem.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**utils**](dir_821002d4f10779a80d4fb17bc32f21f1.md) **>** [**filesystem.h**](filesystem_8h.md)

[Go to the documentation of this file](filesystem_8h.md)


```C++
#pragma once

#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

namespace xt::utils::fs {

    std::size_t countFiles(const std::filesystem::path& path , bool recursive = true);

}
```


