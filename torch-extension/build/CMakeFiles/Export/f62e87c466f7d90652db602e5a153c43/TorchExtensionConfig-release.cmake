#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TorchExtension" for configuration "Release"
set_property(TARGET TorchExtension APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TorchExtension PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libTorchExtension.so.0.1.0"
  IMPORTED_SONAME_RELEASE "libTorchExtension.so.1"
  )

list(APPEND _cmake_import_check_targets TorchExtension )
list(APPEND _cmake_import_check_files_for_TorchExtension "${_IMPORT_PREFIX}/lib/libTorchExtension.so.0.1.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
