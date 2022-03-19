#include <alpaka/alpaka.hpp>

#include "config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct Kernel {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc,
                                  const int* __restrict__ data,
                                  int* __restrict__ result,
                                  const int size) const {
      int i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
      if (i < size)
        alpaka::atomicAdd(acc, result, data[i], alpaka::hierarchy::Blocks{});
    }
  };

  bool enabled() {
    int count = alpaka::getDevCount<Platform>();
    return (count > 0);
  }

  void execute() {
    int size = 10;

    // host "device"
    auto host = alpaka::getDevByIdx<HostPlatform>(0);

    // use the first device
    auto device = alpaka::getDevByIdx<Platform>(0);
    auto queue = Queue{device};
    std::cout << "device: " << alpaka::getName(device) << std::endl;

    auto buffer_h = alpaka::allocBuf<int, Idx>(host, Vec1D{size});
    for (int i = 0; i < size; ++i) {
      buffer_h[i] = 1;
    }
    auto result_h = alpaka::allocBuf<int, Idx>(host, Scalar{});
    *result_h = 0;

    auto buffer = alpaka::allocBuf<int, Idx>(device, Vec1D{size});
    alpaka::memcpy(queue, buffer, buffer_h);
    auto result = alpaka::allocBuf<int, Idx>(device, Scalar{});
    alpaka::memset(queue, result, 0);

    const Vec1D blocks{1};
    const Vec1D threads{32};
    const Vec1D elements{1};
    const WorkDiv1D workDiv{blocks, threads, elements};

    std::cout << "accalerator: " << alpaka::getAccName<Acc1D>() << std::endl;
    auto kernelTask = alpaka::createTaskKernel<Acc1D>(workDiv, Kernel{}, buffer.data(), result.data(), size);
    alpaka::enqueue(queue, kernelTask);
    alpaka::memcpy(queue, result_h, result);
    alpaka::wait(queue);

    std::cout << "result: " << *result_h << std::endl;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
