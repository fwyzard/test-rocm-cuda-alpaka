#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

namespace alpaka_cuda {

  bool enabled();

  void execute();

}  // namespace alpaka_cuda

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

namespace alpaka_rocm {

  bool enabled();

  void execute();

}  // namespace alpaka_rocm

#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
