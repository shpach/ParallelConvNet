#include <stdio.h>
#include <cuda.h>

int query() {
  int deviceCount;

  cudaGetDeviceCount(&deviceCount);

  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, dev);

    if (dev == 0) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
	printf("No GPU detected.\n");
        return -1;
      } else if (deviceCount == 1) {
        //@@ WbLog is a provided logging API (similar to Log4J).
        //@@ The logging function wbLog takes a level which is either
        //@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or TRACE and a
        //@@ message to be printed.
        printf("There is 1 device supporting CUDA\n");
      } else {
        printf("There are %d devices supporting CUDA\n", deviceCount);
      }
    }

    printf("Device %d name: %s\n", dev, deviceProp.name);
    printf("Computational Capabilities: %d %d\n.", deviceProp.major,
          deviceProp.minor);
    printf(" Maximum global memory size: %zd\n",
          deviceProp.totalGlobalMem);
    printf(" Maximum constant memory size: %zd\n",
          deviceProp.totalConstMem);
    printf(" Maximum shared memory size per block: %zd\n",
          deviceProp.sharedMemPerBlock);
    printf(" Maximum block dimensions: %dx%dx%d\n",
          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
          deviceProp.maxThreadsDim[2]);
    printf(" Maximum grid dimensions: %dx%dx%d\n", deviceProp.maxGridSize[0],
          deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf(" Warp size: %d\n", deviceProp.warpSize);
		printf(" Multi processor count: %d\n", deviceProp.multiProcessorCount);
  }
  return 1;
}
