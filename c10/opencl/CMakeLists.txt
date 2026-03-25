# OpenCL/SYCL backend for Intel GPU support
# AI-TRANSLATED: Modeled after c10/cuda/CMakeLists.txt

if(USE_OPENCL)
  file(GLOB C10_OPENCL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  )

  add_library(c10_opencl ${C10_OPENCL_SRCS})
  target_link_libraries(c10_opencl PUBLIC c10)
  target_include_directories(c10_opencl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
  )

  # SYCL TODO: needs_review - Link against Intel oneAPI SYCL runtime
  # find_package(IntelSYCL REQUIRED)
  # target_link_libraries(c10_opencl PUBLIC IntelSYCL::SYCL)

  target_compile_definitions(c10_opencl PRIVATE C10_OPENCL_BUILD_MAIN_LIB)

  install(TARGETS c10_opencl DESTINATION lib)

  # Install headers
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/c10/opencl
    FILES_MATCHING PATTERN "*.h"
  )
endif()
