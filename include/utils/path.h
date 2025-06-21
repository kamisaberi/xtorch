#pragma once
#include <string>
#include <filesystem>
#include <optional> // For get_library_paths return

// Your XT_PUBLIC macro for exporting/importing symbols
#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_XTORCH_DLL // This should be defined by your xTorch library's build
    #define XT_UTILS_PUBLIC __declspec(dllexport)
#else
    #define XT_UTILS_PUBLIC __declspec(dllimport) // For consumers
#endif
#else
#define XT_UTILS_PUBLIC __attribute__ ((visibility ("default")))
#endif

namespace xt
{
    namespace fs = std::filesystem;

    /**
     * @brief Struct to hold essential xTorch installation paths.
     */
    struct XTorchPaths
    {
        fs::path install_prefix;
        fs::path library_path; // Path to libxTorch.so itself
        fs::path share_dir; // <prefix>/share/xtorch
        fs::path venv_dir; // <prefix>/share/xtorch/venv
        fs::path python_executable; // <prefix>/share/xtorch/venv/bin/python
        fs::path conversion_script; // <prefix>/share/xtorch/tools/convert_hf_model.py
        // Add other paths as needed
    };

    /**
     * @brief Retrieves the essential installation paths for the xTorch library.
     * This function is part of libxTorch.so and determines paths relative to itself.
     * @return An optional struct containing paths, or std::nullopt on failure.
     */
    XT_UTILS_PUBLIC std::optional<XTorchPaths> get_internal_library_paths();


    // /**
    //  * @brief Downloads a file from a given URL to a specified destination path.
    //  * This function uses system tools (like curl or wget) or a C++ HTTP library.
    //  * For this example, we'll simulate it with a system call to curl.
    //  *
    //  * @param url The URL of the file to download.
    //  * @param destination_path The full path (including filename) to save the downloaded file.
    //  * @return True on success, false on failure.
    //  */
    // XT_UTILS_PUBLIC bool download_file(const std::string& url, const fs::path& destination_path);

    /**
     * @brief Converts a Hugging Face model to TorchScript using the internal Python script.
     *
     * @param hf_model_name The name of the model on Hugging Face Hub.
     * @param output_torchscript_path The full path to save the converted .pt model.
     * @param batch_size Batch size for dummy input during tracing.
     * @param image_size Image size (height/width) for dummy input.
     * @param channels Number of channels for dummy input.
     * @return True on success, false on failure.
     */
    XT_UTILS_PUBLIC bool convert_hf_model_to_torchscript_from_lib(
        const std::string& hf_model_name,
        const fs::path& output_torchscript_path,
        int batch_size = 1,
        int image_size = 224,
        int channels = 3
    );
} // namespace xtorch
