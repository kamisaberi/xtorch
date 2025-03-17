project(examples)
list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/c++/libtorch/)
find_package(Torch REQUIRED)
#find_package(TorchExtension REQUIRED)
find_package(LibLZMA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_SOURCE_FILES
        ../src/utils/extract.cpp
        ../src/utils/downloader.cpp
        ../src/media/image/processing.cpp
        ../src/datasets/mnist.cpp
        ../src/models/cnn/lenet5.cpp
        ../src/utils/md5.cpp
        ../include/types/arguments.h
        ../src/definitions/transforms.cpp
        ../include/types/enums.h
        ../include/datasets/base.h
        ../src/datasets/base.cpp
        ../include/datasets/functions.h
        ../src/datasets/functions.cpp
        ../include/base/datasets.h
        ../examples/cnn/includes/base.h
)

set(TEST_LIBRARIES ${CURL_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${ZLIB_LIBS}
        ${ZIP_LIBRARIES}
        zip
        ${ZLIB_LIBRARIES}
        ${LIBTAR_LIBRARY}
        OpenSSL::SSL
        OpenSSL::Crypto
        LibLZMA::LibLZMA
)

add_executable(app01 ${TEST_SOURCE_FILES} cnn/lenet5-mnist.cpp)
target_link_libraries(app01 "${TORCH_LIBRARIES}")
set_property(TARGET app01 PROPERTY CXX_STANDARD 17)
target_include_directories(app01 PRIVATE ${LIBTAR_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(app01 ${TEST_LIBRARIES})
