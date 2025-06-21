#include "include/utils/path.h" // The header we just defined
#include <iostream>
#include <cstdlib> // For std::system
#include <vector>  // For potential future use

// For get_this_library_path_internal (dladdr/GetModuleHandleExA)
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace xt
{
    // Internal helper to get the path of libxTorch.so itself
    // This is NOT exported directly.
    std::string get_this_library_path_internal_for_utils()
    {
#ifdef _WIN32
    char path[MAX_PATH];
    HMODULE hm = NULL;
    // Get handle to THIS module (libxTorch.so/dll)
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)&get_this_library_path_internal_for_utils, &hm) == 0) {
        return "";
    }
    if (GetModuleFileNameA(hm, path, sizeof(path)) == 0) {
        return "";
    }
    return std::string(path);
#else
        Dl_info info;
        if (dladdr((void*)&get_this_library_path_internal_for_utils, &info) != 0)
        {
            if (info.dli_fname)
            {
                return std::string(info.dli_fname);
            }
        }
        return "";
#endif
    }


    // Implementation of the exported path finding function
    std::optional<XTorchPaths> get_internal_library_paths()
    {
        XTorchPaths paths;
        fs::path lib_path_fs = get_this_library_path_internal_for_utils();

        if (lib_path_fs.empty())
        {
            std::cerr << "xTorch Utils ERROR: Could not determine own library path." << std::endl;
            return std::nullopt;
        }
        paths.library_path = lib_path_fs;
        paths.install_prefix = lib_path_fs.parent_path().parent_path(); // e.g., from <prefix>/lib/libfoo.so to <prefix>
        paths.share_dir = paths.install_prefix / "share" / "xtorch"; // Assuming "xtorch" subfolder
        paths.venv_dir = paths.share_dir / "venv";
        paths.python_executable = paths.venv_dir / "bin" / "python";
        paths.conversion_script = paths.share_dir / "py_modules" / "convert_hf_model.py"; // Path to your python script

        // Basic check
        if (!fs::exists(paths.venv_dir))
        {
            std::cerr << "xTorch Utils WARNING: Venv directory not found at: " << paths.venv_dir << std::endl;
            // Depending on strictness, you might return nullopt here or let other functions fail
        }
        return paths;
    }


    // Helper to quote paths for command line
    std::string quote_if_needed(const fs::path& p)
    {
        std::string s = p.string();
        if (s.find_first_of(" \t\n\v\f\r\"'") != std::string::npos)
        {
            return "\"" + s + "\"";
        }
        return s;
    }

    // // Implementation of download_file
    // bool download_file(const std::string& url, const fs::path& destination_path)
    // {
    //     std::cout << "[xTorch Utils] Attempting to download: " << url << std::endl;
    //     std::cout << "[xTorch Utils] To destination: " << destination_path << std::endl;
    //
    //     // Ensure parent directory exists
    //     if (destination_path.has_parent_path())
    //     {
    //         fs::create_directories(destination_path.parent_path());
    //     }
    //
    //     // Using curl as an example. You might prefer a C++ HTTP library for robustness.
    //     // Ensure curl is installed on the system where this runs.
    //     std::string command = "curl -L -o " + quote_if_needed(destination_path) + " " + quote_if_needed(url);
    //     // Add --silent or -s for less verbose curl output: "curl -s -L -o ..."
    //
    //     std::cout << "[xTorch Utils] Executing download command: " << command << std::endl;
    //     int result = std::system(command.c_str());
    //
    //     if (result == 0 && fs::exists(destination_path))
    //     {
    //         std::cout << "[xTorch Utils] Download successful." << std::endl;
    //         return true;
    //     }
    //     else
    //     {
    //         std::cerr << "[xTorch Utils] Download failed. Curl exit code: " << result << std::endl;
    //         if (fs::exists(destination_path))
    //         {
    //             // Clean up partial download
    //             fs::remove(destination_path);
    //         }
    //         return false;
    //     }
    // }

    // Implementation of convert_hf_model_to_torchscript_from_lib
    bool convert_hf_model_to_torchscript_from_lib(
        const std::string& hf_model_name,
        const fs::path& output_torchscript_path,
        int batch_size,
        int image_size,
        int channels)
    {
        auto paths_opt = get_internal_library_paths();
        if (!paths_opt)
        {
            std::cerr << "xTorch Utils ERROR: Cannot run model conversion, library paths unknown." << std::endl;
            return false;
        }
        const XTorchPaths& paths = *paths_opt;

        if (!fs::exists(paths.python_executable))
        {
            std::cerr << "xTorch Utils ERROR: Python executable not found at: " << paths.python_executable << std::endl;
            return false;
        }
        if (!fs::exists(paths.conversion_script))
        {
            std::cerr << "xTorch Utils ERROR: Python conversion script not found at: " << paths.conversion_script <<
                std::endl;
            return false;
        }

        std::string command = quote_if_needed(paths.python_executable) + " " +
            quote_if_needed(paths.conversion_script) + " " +
            quote_if_needed(output_torchscript_path) +
            " --model_name " + quote_if_needed(hf_model_name) +
            " --batch_size " + std::to_string(batch_size) +
            " --image_size " + std::to_string(image_size) +
            " --channels " + std::to_string(channels);

        std::cout << "[xTorch Utils] Executing model conversion command: " << command << std::endl;
        int result = std::system(command.c_str());

        if (result == 0)
        {
            std::cout << "[xTorch Utils] Python model conversion script executed successfully (or reported success)." <<
                std::endl;
            return true;
        }
        else
        {
            std::cerr << "[xTorch Utils] Python model conversion script failed or reported error. Exit code: " << result
                << std::endl;
            return false;
        }
    }
} // namespace xtorch
