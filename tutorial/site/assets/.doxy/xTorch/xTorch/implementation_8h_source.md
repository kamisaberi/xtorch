

# File implementation.h

[**File List**](files.md) **>** [**exceptions**](dir_06ec884a0825782b323e4577406ae7aa.md) **>** [**implementation.h**](implementation_8h.md)

[Go to the documentation of this file](implementation_8h.md)


```C++
#pragma once

#include <iostream>


class NotImplementedException : public std::logic_error {
public:
    NotImplementedException() : std::logic_error("Function not yet implemented") {};
};
```


