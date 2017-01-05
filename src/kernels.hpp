__global__ void conv_forward_para(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]);
__global__ void average_pool_para(const float *X, const int xdims[4],
                          float *Y);
__global__ void conv_forward_para_v2(const float *X, const int xdims[4], const float *W, const int wdims[4], float *Y, const int ydims[4]);
__global__ void relu_para(float *X, const int xdims[4]);
__global__ void fully_forward_p(const float *X, const int xdims[2], float *W, const int wdims[2], float *Y, const int ydims[2]);
