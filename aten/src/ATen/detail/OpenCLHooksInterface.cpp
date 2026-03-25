// AI-TRANSLATED: OpenCL hooks interface implementation
#include <ATen/detail/OpenCLHooksInterface.h>

namespace at {
namespace detail {

const OpenCLHooksInterface& getOpenCLHooks() {
  static std::unique_ptr<OpenCLHooksInterface> opencl_hooks;
#if !defined(USE_OPENCL)
  // If not built with OpenCL, return a default (stub) hooks object
  static OpenCLHooksInterface default_hooks;
  return default_hooks;
#else
  auto creator = OpenCLHooksRegistry()->Create("OpenCLHooks", OpenCLHooksArgs{});
  if (creator) {
    opencl_hooks.reset(creator);
  } else {
    opencl_hooks = std::make_unique<OpenCLHooksInterface>();
  }
  return *opencl_hooks;
#endif
}

} // namespace detail

C10_DEFINE_REGISTRY(OpenCLHooksRegistry, OpenCLHooksInterface, OpenCLHooksArgs)

} // namespace at
