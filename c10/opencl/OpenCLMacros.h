// AI-TRANSLATED: OpenCL backend macros for Intel GPU support
#pragma once

#ifdef USE_OPENCL

#define C10_OPENCL_API C10_EXPORT
#define C10_OPENCL_BUILD_MAIN_LIB

#else

#define C10_OPENCL_API C10_IMPORT

#endif
