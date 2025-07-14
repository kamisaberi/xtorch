add_executable(xtorch_tests
        test_linalg.cpp
)

# Link our test executable against our main library (xtorch)
# and the Google Test framework.
# NOTE: "gtest_main" includes the main() function for the test runner,
# so you don't have to write it yourself.
target_link_libraries(xtorch_tests PRIVATE
        xTorch      # Your library target name, adjust if different
        gtest_main
)

# Include the gtest headers
target_include_directories(xtorch_tests PRIVATE
        ../include   # Allow tests to #include "math/linalg.h"
)

# Tell CTest about our test executable so we can run it with the "ctest" command
include(GoogleTest)
gtest_add_tests(TARGET xtorch_tests)