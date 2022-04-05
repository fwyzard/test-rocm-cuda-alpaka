#include <cstring>
#include <iostream>

#include "backend.h"

enum Status { uninitialised, requested, initialised, failed };

int main(int argc, const char** argv) {
  Status status_rocm = uninitialised;
  Status status_cuda = uninitialised;
  Status status_serial = uninitialised;

  // If there are no command line options,
  // request all backends available at compile time
  if (argc == 1) {
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    status_rocm = requested;
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    status_cuda = requested;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    status_serial = requested;
#endif
  }

  // Parse the command line options
  for (int i = 1; i < argc; ++i) {
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    // HIP with ROCm backend
    if (std::strcmp(argv[i], "--rocm") == 0) {
      status_rocm = requested;
      continue;
    }
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    // CUDA backend
    if (std::strcmp(argv[i], "--cuda") == 0) {
      status_cuda = requested;
      continue;
    }
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    // CPU-only serial backend
    if (std::strcmp(argv[i], "--serial") == 0) {
      status_serial = requested;
      continue;
    }
#endif

    if (std::strcmp(argv[i], "--all") == 0) {
      // request all backends available at compile time
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
      status_rocm = requested;
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      status_cuda = requested;
#endif
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
      status_serial = requested;
#endif
    } else {
      // unknown option
      std::cout << std::endl;
      std::cout << argv[0] << ": unrecognised option " << argv[i] << std::endl;
    }
  }

  // Try to initialise the requested backends
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  // HIP with ROCm backend
  if (status_rocm == requested) {
    if (alpaka_rocm::enabled()) {
      std::cout << "ROCm runtime initialised" << std::endl;
      status_rocm = initialised;
    } else {
      std::cout << "Failed to initialise the ROCm runtime" << std::endl;
      status_rocm = failed;
    }
  }
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  // CUDA backend
  if (status_cuda == requested) {
    if (alpaka_cuda::enabled()) {
      std::cout << "CUDA runtime initialised" << std::endl;
      status_cuda = initialised;
    } else {
      std::cout << "Failed to initialise the CUDA runtime" << std::endl;
      status_cuda = failed;
    }
  }
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // CPU-only serial backend
  if (status_serial == requested) {
    if (alpaka_serial::enabled()) {
      std::cout << "CPU-only serial runtime initialised" << std::endl;
      status_serial = initialised;
    } else {
      std::cout << "Failed to initialise the CPU-only serial runtime" << std::endl;
      status_serial = failed;
    }
  }
#endif

  // Submit work to the enbled backends
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
  // HIP with ROCm backend
  if (status_rocm == initialised) {
    std::cout << std::endl;
    alpaka_rocm::execute();
    std::cout << "ROCm kernel ran successfully" << std::endl;
  }
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  // CUDA backend
  if (status_cuda == initialised) {
    std::cout << std::endl;
    alpaka_cuda::execute();
    std::cout << "CUDA kernel ran successfully" << std::endl;
  }
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // CPU-only serial backend
  if (status_serial == initialised) {
    std::cout << std::endl;
    alpaka_serial::execute();
    std::cout << "CPU-only serial kernel ran successfully" << std::endl;
  }
#endif

  return 0;
}
