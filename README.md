# Test application using both CUDA and HIP/ROCm, via Alpaka

### Clone the repository
```bash
git clone git@github.com:fwyzard/test-rocm-cuda-alpaka.git
```

### Build the application
```bash
cd test-rocm-cuda-alpaka
make
```

### Running using a single backend
On a machine with a GeForce GTX 1080 Ti:
```bash
$ ./test-cuda --cuda
NVIDIA GeForce GTX 1080 Ti
result: 10
CUDA kernel ran successfully
```

On a machine with a Radeon Pro WX 9100:
```bash
 ./test-rocm --rocm
Radeon Pro WX 9100
result: 10
ROCm kernel ran successfully
```

### Running using both backends
```bash
./test --rocm --cuda
```

On a machine without any NVIDIA or AMD GPU, the expected output is
```
Failed to initialise the ROCm runtime
Failed to initialise the CUDA runtime
```

On a machine with an NVIDIA GPU and the CUDA runtime, the expected output is something like
```
Failed to initialise the ROCm runtime
NVIDIA GeForce GTX 1080 Ti
result: 10
CUDA kernel ran successfully
```

On a machine with an AMD GPU and the HIP/ROCm runtime, the expected output is something like
```
Radeon Pro WX 9100
result: 10
ROCm kernel ran successfully
Failed to initialise the CUDA runtime
```

On a machine with both an NVIDIA and an AMD GPU to run the test, the expected output is something like
```
Radeon Pro WX 9100
result: 10
ROCm kernel ran successfully
NVIDIA GeForce GTX 1080 Ti
result: 10
CUDA kernel ran successfully
```
