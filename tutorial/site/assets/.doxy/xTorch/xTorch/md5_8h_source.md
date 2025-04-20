

# File md5.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**utils**](dir_821002d4f10779a80d4fb17bc32f21f1.md) **>** [**md5.h**](md5_8h.md)

[Go to the documentation of this file](md5_8h.md)


```C++
#pragma once

#include <iostream>
#include <fstream>
//#include <openssl/md5.h>
#include <iomanip>
#include <openssl/evp.h>
#include <openssl/err.h>

namespace xt::utils {
    std::string get_md5_checksum(const std::string &filename);
}
```


