// AI-TRANSLATED: OpenCL device guard for Intel GPU support
#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/opencl/OpenCLMacros.h>
#include <c10/opencl/impl/OpenCLGuardImpl.h>

namespace c10::opencl {

using OpenCLGuard = c10::impl::InlineDeviceGuard<impl::OpenCLGuardImpl>;
using OptionalOpenCLGuard = c10::impl::InlineOptionalDeviceGuard<impl::OpenCLGuardImpl>;
using OpenCLStreamGuard = c10::impl::InlineStreamGuard<impl::OpenCLGuardImpl>;
using OptionalOpenCLStreamGuard = c10::impl::InlineOptionalStreamGuard<impl::OpenCLGuardImpl>;

} // namespace c10::opencl
