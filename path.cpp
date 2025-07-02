// This is the ONLY C++ code needed in the xTorch library itself for this approach.
// It provides a stable address within libxTorch.so that our header can use.

// Make sure XT_PUBLIC is defined correctly for export
#if defined _WIN32 || defined __CYGWIN__
  #define XT_ANCHOR_EXPORT __declspec(dllexport)
#else
  #define XT_ANCHOR_EXPORT __attribute__ ((visibility ("default")))
#endif

extern "C" XT_ANCHOR_EXPORT void xtorch_internal_anchor_function() {
    // This function does nothing. Its sole purpose is to exist at a known
    // address within libxTorch.so, so dladdr can find the library.
}