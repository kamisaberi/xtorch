# cmake/installation.cmake
# ===============================================================
#  Defines the installation rules for the xTorch project.
# ===============================================================

include(GNUInstallDirs) # Standard GNU installation directories

# --- Install Targets ---
install(TARGETS xTorch EXPORT xTorchTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# --- Install Headers and Other Project Files ---
string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LOWER)
install(FILES xtorch.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
install(DIRECTORY third_party DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME_LOWER})
install(DIRECTORY python/xtorch_py/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME_LOWER}/py_modules)
install(FILES python/requirements.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME_LOWER})

# --- Install CMake Package Configuration Files ---
install(EXPORT xTorchTargets
        FILE xTorchTargets.cmake
        NAMESPACE xTorch::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xTorch)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/xTorchConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xTorch)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/xTorchConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xTorch)

# --- Add Uninstall Target ---
if (NOT TARGET uninstall)
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
            IMMEDIATE @ONLY)
    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()