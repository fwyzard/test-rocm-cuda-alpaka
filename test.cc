#include <cstring>
#include <iostream>

#include "backend.h"

int main(int argc, const char** argv) {
  for (int i = 1; i < argc; ++i) {
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    // HIP with ROCm backend
    if (std::strcmp(argv[i], "--rocm") == 0) {
      std::cout << std::endl;
      if (alpaka_rocm::enabled()) {
        alpaka_rocm::execute();
        std::cout << "ROCm kernel ran successfully" << std::endl;
      } else {
        std::cout << "Failed to initialise the ROCm runtime" << std::endl;
      }
    } else
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    // CUDA backend
    if (std::strcmp(argv[i], "--cuda") == 0) {
      std::cout << std::endl;
      if (alpaka_cuda::enabled()) {
        alpaka_cuda::execute();
        std::cout << "CUDA kernel ran successfully" << std::endl;
      } else {
        std::cout << "Failed to initialise the CUDA runtime" << std::endl;
      }
    } else
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    // CPU-only serial backend
    if (std::strcmp(argv[i], "--serial") == 0) {
      std::cout << std::endl;
      if (alpaka_serial::enabled()) {
        alpaka_serial::execute();
        std::cout << "CPU-only serial kernel ran successfully" << std::endl;
      } else {
        std::cout << "Failed to initialise the CPU-only serial runtime" << std::endl;
      }
    } else
#endif

    {
      // unknown option
      std::cout << std::endl;
      std::cout << argv[0] << ": unrecognised option " << argv[i] << std::endl;
    }
  }

  return 0;
}
