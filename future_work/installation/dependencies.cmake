# cmake/dependencies.cmake
# ===============================================================
#  Downloads and extracts external dependencies.
#  This script expects LIBTORCH_URL, ONNXRUNTIME_URL, etc. to
#  be set by the calling platform script.
# ===============================================================

set(DEPS_DIR "${CMAKE_SOURCE_DIR}/third_party")
set(LIBTORCH_DIR "${DEPS_DIR}/libtorch")
set(ONNXRUNTIME_DIR "${DEPS_DIR}/onnxruntime")

# Create the third_party directory if it doesn't exist
if (NOT EXISTS "${DEPS_DIR}")
    file(MAKE_DIRECTORY "${DEPS_DIR}")
endif ()

# --- Fetch LibTorch if it doesn't exist locally ---
if (NOT EXISTS "${LIBTORCH_DIR}")
    message(STATUS "LibTorch not found locally. Downloading and extracting...")
    string(REGEX MATCH "([^/]+)$" LIBTORCH_FILENAME ${LIBTORCH_URL})
    set(LIBTORCH_ARCHIVE "${DEPS_DIR}/${LIBTORCH_FILENAME}")
    file(DOWNLOAD ${LIBTORCH_URL} ${LIBTORCH_ARCHIVE} EXPECTED_HASH SHA256=${LIBTORCH_SHA256} SHOW_PROGRESS)
    message(STATUS "Extracting LibTorch...")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${LIBTORCH_ARCHIVE} WORKING_DIRECTORY ${DEPS_DIR})
    file(REMOVE ${LIBTORCH_ARCHIVE})
    message(STATUS "LibTorch setup complete.")
else()
    message(STATUS "Found local LibTorch at ${LIBTORCH_DIR}")
endif()

# --- Fetch ONNX Runtime if it doesn't exist locally ---
if (NOT EXISTS "${ONNXRUNTIME_DIR}")
    message(STATUS "ONNX Runtime not found locally. Downloading and extracting...")
    string(REGEX MATCH "([^/]+)$" ONNXRUNTIME_FILENAME ${ONNXRUNTIME_URL})
    set(ONNXRUNTIME_ARCHIVE "${DEPS_DIR}/${ONNXRUNTIME_FILENAME}")
    file(DOWNLOAD ${ONNXRUNTIME_URL} ${ONNXRUNTIME_ARCHIVE} EXPECTED_HASH SHA256=${ONNXRUNTIME_SHA256} SHOW_PROGRESS)
    message(STATUS "Extracting ONNX Runtime...")
    set(ONNXRUNTIME_TEMP_DIR "${DEPS_DIR}/onnxruntime_temp")
    file(MAKE_DIRECTORY ${ONNXRUNTIME_TEMP_DIR})
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${ONNXRUNTIME_ARCHIVE} WORKING_DIRECTORY ${ONNXRUNTIME_TEMP_DIR})
    file(GLOB EXTRACTED_DIR "${ONNXRUNTIME_TEMP_DIR}/*")
    file(RENAME ${EXTRACTED_DIR} ${ONNXRUNTIME_DIR})
    file(REMOVE_RECURSE ${ONNXRUNTIME_TEMP_DIR})
    file(REMOVE ${ONNXRUNTIME_ARCHIVE})
    message(STATUS "ONNX Runtime setup complete.")
else()
    message(STATUS "Found local ONNX Runtime at ${ONNXRUNTIME_DIR}")
endif()

# --- Configure the downloaded libraries ---
set(CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
find_package(Torch REQUIRED)
message(STATUS "Configured LibTorch: ${TORCH_VERSION}")

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" REQUIRED)
message(STATUS "Configured ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")