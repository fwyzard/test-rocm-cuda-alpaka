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

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

namespace alpaka_serial {

  bool enabled();

  void execute();

}  // namespace alpaka_serial

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
