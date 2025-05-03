/**
 * @file utils.h
 * @brief Master include file for XT utility library
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 * @date Created: [Current Date]
 * @version 1.0
 *
 * This is the main header file that includes all utility components
 * from the XT library. It serves as a single include point for
 * accessing all utility functionality.
 */

#pragma once

// Core utility components
#include "base.h"        // Basic utility functions and definitions

// Network and file operations
#include "downloader.h"  // File downloading utilities

// Archive operations
#include "extract.h"     // File extraction utilities

// Filesystem operations
#include "filesystem.h"  // Filesystem utility functions

// Cryptographic utilities
#include "md5.h"         // MD5 checksum calculation

// String manipulation
#include "string.h"      // String utility functions

/*
 * Author: Kamran Saberifard
 * Email: kamisaberi@gmail.com
 * GitHub: https://github.com/kamisaberi
 *
 * This implementation is part of the XT Utility Library.
 * Copyright (c) 2023 Kamran Saberifard. All rights reserved.
 *
 * Licensed under the MIT License. See LICENSE file in the project root
 * for full license information.
 */

/**
 * @namespace xt::utils
 * @brief Main namespace for all utility functions
 *
 * The xt::utils namespace contains all utility functions organized into
 * logical components. This header provides
 */