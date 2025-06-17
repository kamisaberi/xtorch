#pragma once


#include <string>
#include <filesystem>
#include <optional>
#include <iostream> // For error messages

// Platform-specific includes for dladdr / GetModuleHandleExA
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace xt
{
    namespace fs = std::filesystem;

    // Forward declaration of the anchor function from libxTorch.so
    // The consumer will link against libxTorch.so to resolve this.
#if defined _WIN32 || defined __CYGWIN__
  #define XT_ANCHOR_IMPORT __declspec(dllimport)
#else
#define XT_ANCHOR_IMPORT // No special attribute needed for import on Linux/macOS usually
#endif

    extern "C" XT_ANCHOR_IMPORT void xtorch_internal_anchor_function();


    /**
     * @brief A struct to hold all the important paths for the xTorch library.
     */
    struct XTorchPaths
    {
        fs::path install_prefix;
        fs::path library_path;
        fs::path share_dir;
        fs::path venv_dir;
        fs::path python_executable;
        fs::path python_modules_dir;
    };

    /**
     * @brief Locates and returns all essential xTorch paths.
     *
     * This function is inline and compiled as part of the consumer's code.
     * It uses the address of `xtorch_internal_anchor_function` (which is in
     * libxTorch.so) to determine the path to libxTorch.so.
     *
     * @return An std::optional containing the XTorchPaths struct.
     *         The optional will be empty (std::nullopt) if paths could not be determined.
     */
    inline std::optional<XTorchPaths> get_library_paths()
    {
        XTorchPaths paths_struct;
        fs::path lib_xtorch_actual_path;

        // Use the address of the anchor function to find libxTorch.so
#ifdef _WIN32
    char path_buf[MAX_PATH];
    HMODULE hm = NULL;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)&xtorch_internal_anchor_function, &hm) == 0) {
        std::cerr << "xTorch ERROR (in header): GetModuleHandleExA failed for anchor." << std::endl;
        return std::nullopt;
    }
    if (GetModuleFileNameA(hm, path_buf, sizeof(path_buf)) == 0) {
        std::cerr << "xTorch ERROR (in header): GetModuleFileNameA failed for anchor." << std::endl;
        return std::nullopt;
    }
    lib_xtorch_actual_path = std::string(path_buf);
#else
        Dl_info info;
        if (dladdr((void*)&xtorch_internal_anchor_function, &info) != 0)
        {
            if (info.dli_fname)
            {
                lib_xtorch_actual_path = std::string(info.dli_fname);
            }
            else
            {
                std::cerr << "xTorch ERROR (in header): dladdr returned null for dli_fname." << std::endl;
                return std::nullopt;
            }
        }
        else
        {
            std::cerr << "xTorch ERROR (in header): dladdr failed for anchor function. Error: " << dlerror() <<
                std::endl;
            return std::nullopt;
        }
#endif

        if (lib_xtorch_actual_path.empty())
        {
            std::cerr << "xTorch ERROR (in header): Could not determine path to libxTorch.so via anchor." << std::endl;
            return std::nullopt;
        }
        paths_struct.library_path = lib_xtorch_actual_path;

        // Now derive all other paths from the location of libxTorch.so
        paths_struct.install_prefix = lib_xtorch_actual_path.parent_path().parent_path();
        paths_struct.share_dir = paths_struct.install_prefix / "share" / "xtorch"; // Assuming "xtorch" as subfolder
        paths_struct.venv_dir = paths_struct.share_dir / "venv";
        paths_struct.python_executable = paths_struct.venv_dir / "bin" / "python";
        paths_struct.python_modules_dir = paths_struct.share_dir / "py_modules";

        if (!fs::exists(paths_struct.venv_dir))
        {
            std::cerr << "xTorch WARNING (in header): Venv directory not found at expected location: "
                << paths_struct.venv_dir << std::endl;
            // Decide if this is a fatal error for returning std::nullopt
            // For now, let's say it is.
            return std::nullopt;
        }

        return paths_struct;
    }
} // namespace xtorch
