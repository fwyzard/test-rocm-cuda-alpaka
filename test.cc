#include <cstring>
#include <iostream>

#include "backend.h"

int main(int argc, const char** argv) {
  for (int i = 1; i < argc; ++i) {

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    // HIP with ROCm backend
    if (std::strcmp(argv[i], "--rocm") == 0) {
      if (alpaka_rocm::enabled()) {
        alpaka_rocm::execute();
        std::cout << "ROCm kernel ran successfully" << std::endl;
      } else {
        std::cout << "Failed to initialise the ROCm runtime" << std::endl;
      }
      std::cout << std::endl;
    } else
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    // CUDA backend
    if (std::strcmp(argv[i], "--cuda") == 0) {
      if (alpaka_cuda::enabled()) {
        alpaka_cuda::execute();
        std::cout << "CUDA kernel ran successfully" << std::endl;
      } else {
        std::cout << "Failed to initialise the CUDA runtime" << std::endl;
      }
      std::cout << std::endl;
    } else
#endif
    {
      // unknown option
      std::cout << argv[0] << ": unrecognised option " << argv[i] << std::endl;
    }
  }

  return 0;
}
