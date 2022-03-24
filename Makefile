.PHONY: all clean

COMMA=,

CXX := g++
CXX_FLAGS := -std=c++17 -O2 -g -Ialpaka/include -Iboost/include -DALPAKA_DEBUG=0
CXX_HOST_FLAGS := -Wall -Wno-unused-result -fPIC

HIP_BASE := /cvmfs/patatrack.cern.ch/externals/x86_64/rhel8/amd/rocm-5.0.2
HIPCC := $(HIP_BASE)/hip/bin/hipcc --rocm-path=$(HIP_BASE)
HIP_CXXFLAGS := -I$(HIP_BASE)/include -I$(HIP_BASE)/hiprand/include -I$(HIP_BASE)/rocrand/include
HIP_LDFLAGS := -L$(HIP_BASE)/lib -L$(HIP_BASE)/lib64 -lamdhip64 -lamd_comgr -lhsa-runtime64 -Wl,-rpath,$(HIP_BASE)/lib -Wl,-rpath,$(HIP_BASE)/lib64

CUDA_BASE := /cvmfs/patatrack.cern.ch/externals/x86_64/rhel8/nvidia/cuda-11.5.2
CUDA_DRIVER := /cvmfs/patatrack.cern.ch/externals/x86_64/rhel8/nvidia/compat/510.47.03/
NVCC := $(CUDA_BASE)/bin/nvcc -ccbin ${CXX}
CUDA_CXXFLAGS := -I$(CUDA_BASE)/include --expt-relaxed-constexpr
CUDA_LDFLAGS := -L$(CUDA_BASE)/lib64 -L$(CUDA_DRIVER) -lcudart -lcuda
CUDA_HOST_LDFLAGS := -Wl,-rpath,$(CUDA_BASE)/lib64 -Wl,-rpath,$(CUDA_DRIVER)


all: test test.static 

clean:
	rm -f test test.static *.o *.so

alpaka:
	@mkdir -p alpaka
	git clone git@github.com:fwyzard/alpaka.git -b split_CUDA_ROCm_types

# compile for the HIP/ROCm backend
backend-rocm.o: backend.cc alpaka
	$(HIPCC) $(CXX_FLAGS) $(CXX_HOST_FLAGS) $(HIP_CXXFLAGS) -DALPAKA_ACC_GPU_HIP_ENABLED $< -c -o $@

libbackend-rocm.so: backend-rocm.o
	$(HIPCC) $(CXX_FLAGS) $(CXX_HOST_FLAGS) $(HIP_LDFLAGS) -DALPAKA_ACC_GPU_HIP_ENABLED $< -shared -o $@

# compile for the CUDA backend
backend-cuda.o: backend.cc alpaka
	$(NVCC) $(CXX_FLAGS) -Xcompiler "$(CXX_HOST_FLAGS)" $(CUDA_CXXFLAGS) -DALPAKA_ACC_GPU_CUDA_ENABLED $< -x cu -c -o $@

libbackend-cuda.so: backend-cuda.o
	$(NVCC) $(CXX_FLAGS) -Xcompiler "$(CXX_HOST_FLAGS) $(subst $(COMMA),\$(COMMA),$(CUDA_HOST_LDFLAGS))" $(CUDA_LDFLAGS) -DALPAKA_ACC_GPU_CUDA_ENABLED $< -shared -o $@

# compile for the CPU-only serial backend
backend-serial.o: backend.cc alpaka
	$(CXX) $(CXX_FLAGS) $(CXX_HOST_FLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $< -c -o $@

libbackend-serial.so: backend-serial.o
	$(CXX) $(CXX_FLAGS) $(CXX_HOST_FLAGS) -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED $< -shared -o $@

# main application
test.o: test.cc
	$(CXX) $(CXX_FLAGS) $(CXX_HOST_FLAGS) -DALPAKA_ACC_GPU_HIP_ENABLED -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -c test.cc -o $@

# link all backends
test: test.o libbackend-rocm.so libbackend-cuda.so libbackend-serial.so
	$(CXX) $(CXX_FLAGS) $(CXX_HOST_FLAGS) -o $@ $< -L. -Wl,-rpath,. -lbackend-rocm -lbackend-cuda -lbackend-serial

test.static: test.o backend-rocm.o backend-cuda.o backend-serial.o
	$(CXX) $(CXX_FLAGS) $(CXX_HOST_FLAGS) $^ $(CUDA_LDFLAGS) $(CUDA_HOST_LDFLAGS) $(HIP_LDFLAGS) -o $@
