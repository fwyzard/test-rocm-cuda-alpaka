#pragma once

#include <alpaka/alpaka.hpp>

// common types and dimensions
using Idx = uint32_t;
using Extent = uint32_t;

using Dim0D = alpaka::DimInt<0u>;
using Dim1D = alpaka::DimInt<1u>;

template <typename TDim>
using Vec = alpaka::Vec<TDim, Idx>;
using Scalar = Vec<Dim0D>;
using Vec1D = Vec<Dim1D>;

template <typename TDim>
using WorkDiv = alpaka::WorkDivMembers<TDim, Idx>;
using WorkDiv1D = WorkDiv<Dim1D>;

// host types
using HostDevice = alpaka::DevCpu;
using HostPlatform = alpaka::Pltf<HostDevice>;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_cuda

namespace alpaka_cuda {

  // device types
  using Platform = alpaka::PltfCudaRt;
  using Device = alpaka::DevCudaRt;
  using Queue = alpaka::QueueCudaRtNonBlocking;
  using Event = alpaka::EventCudaRt;

  // accelerator types
  template <typename TDim>
  using Acc = alpaka::AccGpuCudaRt<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;

}  // namespace alpaka_cuda

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_rocm

namespace alpaka_rocm {

  // device types
  using Platform = alpaka::PltfHipRt;
  using Device = alpaka::DevHipRt;
  using Queue = alpaka::QueueHipRtNonBlocking;
  using Event = alpaka::EventHipRt;

  // accelerator types
  template <typename TDim>
  using Acc = alpaka::AccGpuHipRt<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;

}  // namespace alpaka_rocm

#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
