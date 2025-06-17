
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib> // For std::system
#include <filesystem> // For path manipulation

#include "xtorch.h"


// This assumes you have the get_library_paths() function available
// If it's in the header-only xtorch.h:
// #include "xtorch.h" // (or the specific header name)
// If it's exported from the library (the pure modern C++ approach):
// it's already declared in xtorch.hpp

namespace xtorch {

// Helper function to quote paths if they contain spaces
std::string quote_path(const std::filesystem::path& p) {
    return "\"" + p.string() + "\"";
}

bool convert_huggingface_model_to_torchscript(
    const std::string& hf_model_name,
    const std::string& output_torchscript_path,
    int batch_size,
    int image_size,
    int channels) {

    // 1. Get the necessary paths using your library's path-finding mechanism
    std::optional<XTorchPaths> paths_opt = xt::get_library_paths(); // Call your path function

    if (!paths_opt) {
        std::cerr << "xTorch ERROR [Converter]: Could not determine library paths. Cannot run Python script." << std::endl;
        return false;
    }
    const XTorchPaths& paths = *paths_opt;

    // 2. Construct paths to the Python interpreter and the conversion script
    fs::path python_exe = paths.python_executable;
    fs::path conversion_script = paths.share_dir / "tools" / "convert_hf_model.py";

    // 3. Check if the script and python executable exist
    if (!fs::exists(python_exe)) {
        std::cerr << "xTorch ERROR [Converter]: Python interpreter not found at: " << python_exe << std::endl;
        return false;
    }
    if (!fs::exists(conversion_script)) {
        std::cerr << "xTorch ERROR [Converter]: Model conversion script not found at: " << conversion_script << std::endl;
        return false;
    }

    // 4. Construct the command to execute
    //    python /path/to/venv/python /path/to/script.py <output_path_arg> --model_name <hf_model_name_arg> ...
    std::string command = quote_path(python_exe) + " " +
                          quote_path(conversion_script) + " " +
                          quote_path(fs::path(output_torchscript_path)) + // Positional argument
                          " --model_name " + quote_path(fs::path(hf_model_name)) +
                          " --batch_size " + std::to_string(batch_size) +
                          " --image_size " + std::to_string(image_size) +
                          " --channels " + std::to_string(channels);

    std::cout << "xTorch INFO [Converter]: Executing command: " << command << std::endl;

    // 5. Execute the command
    int return_code = std::system(command.c_str());

    if (return_code == 0) {
        std::cout << "xTorch INFO [Converter]: Python script executed successfully." << std::endl;
        return true;
    } else {
        std::cerr << "xTorch ERROR [Converter]: Python script failed with exit code: " << return_code << std::endl;
        // The Python script itself prints detailed errors to stderr, which will also be visible.
        return false;
    }
}

} // namespace xtorch