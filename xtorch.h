/**
 * @file include.h
 * @brief Main include file that aggregates all core components of the framework
 *
 * This header serves as the central include point for all major components
 * of the framework. It provides forward includes for data handling, models,
 * training, transformations, and utility functions.
 *
 * @author Kamran Saberifard
 * @email kamisaberi@gmail.com
 * @github https://github.com/kamisaberi
 *
 * @defgroup CoreComponents Core Framework Components
 * @{
 *   @defgroup DataLoading Data Loading and Processing
 *   @defgroup Datasets Dataset Implementations
 *   @defgroup Media Media Handling
 *   @defgroup Models Model Architectures
 *   @defgroup Training Training Algorithms
 *   @defgroup Transforms Data Transformations
 *   @defgroup Types Core Types
 *   @defgroup Utilities Utility Functions
 *   @defgroup Temp Temporary Components
 * @}
 *
 * @mainpage Framework Overview
 * This framework provides a comprehensive suite of tools for machine learning
 * and data processing. The core components are organized into logical modules
 * that can be included individually or through this master include file.
 *
 * @note For most applications, including this single header is sufficient to
 * access all framework functionality.
 */
#pragma once

#include "include/data_loaders/data_loaders.h"
#include "include/data_parallels/data_parallels.h"
#include "include/datasets/datasets.h"
#include "include/media/media.h"
#include "include/models/models.h"
#include "include/trainers/trainers.h"
#include "include/transforms/transforms.h"
#include "include/types/types.h"
#include "include/utils/utils.h"
#include "include/temp/temp.h"