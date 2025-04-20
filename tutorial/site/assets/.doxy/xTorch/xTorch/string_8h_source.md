

# File string.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**utils**](dir_821002d4f10779a80d4fb17bc32f21f1.md) **>** [**string.h**](string_8h.md)

[Go to the documentation of this file](string_8h.md)


```C++
#pragma once
#include <iostream>
//#include <string>
#include <sstream>
#include <vector>
using namespace std;
namespace xt::utils::string {
    vector<std::string> split(const std::string& str, const std::string& delim);
    std::string trim(std::string& str);

}


```


