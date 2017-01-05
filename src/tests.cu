
static void conv_forward_valid(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]);

void testConv_v2(float* x, float* conv1, const int* dim1, const int* dim2) {
	int *d_xdims, *d_adims, *d_conv1dims;
	float *d_x, *d_conv1;
	float* ad; //will store the result of para conv
  const int adims[] = {dim1[0], (dim1[1] - dim2[0] + 1), (dim1[2] - dim2[1] + 1), dim2[3]};
  auto a = zeros<float>(adims);
  auto ah = zeros<float>(adims); //host
	cudaMalloc(&d_xdims, sizeof(int) * 4);
	cudaMalloc(&d_adims, sizeof(int) * 4);
	cudaMalloc(&d_conv1dims, sizeof(int) * 4);
	cudaMalloc(&d_x, sizeof(float) * dim1[0] * dim1[1] * dim1[2] * dim1[3]);
	cudaMalloc(&d_conv1, sizeof(float) * dim2[0] * dim2[1] * dim2[2] * dim2[3]);
	cudaMalloc(&ad, sizeof(float) * adims[0] * adims[1] * adims[2] * adims[3]);
	cudaMemcpy(d_x, x, sizeof(float) * dim1[0] * dim1[1] * dim1[2] * dim1[3], cudaMemcpyHostToDevice);
	cudaMemcpy(d_conv1, conv1, sizeof(float) * dim2[0] * dim2[1] * dim2[2] * dim2[3], cudaMemcpyHostToDevice);
	cudaMemcpy(d_adims, adims, sizeof(int) * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(d_xdims, dim1, sizeof(int) * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(d_conv1dims, dim2, sizeof(int) * 4, cudaMemcpyHostToDevice);
	cudaMemset(ad, 0, sizeof(float)* adims[0] * adims[1] * adims[2] * adims[3]);

  conv_forward_valid(x, dim1, conv1, dim2, a, adims);

	dim3 convGrid(dim1[0], 1, 1);
	dim3 convBlock(128, 1, 1);
	conv_forward_para_v2 <<<convGrid, convBlock>>> (d_x, d_xdims, 
				d_conv1, d_conv1dims, ad, d_adims);
	cudaDeviceSynchronize();
	cudaMemcpy(ah, ad, sizeof(float) * adims[0] * adims[1] * adims[2] * adims[3], cudaMemcpyDeviceToHost);

	float count = 0.0;
	for (const auto j : range(0, adims[0]*adims[1]*adims[2]*adims[3])) {
			if ((ah[j] - a[j])/a[j] > 1e-2) {
				printf("I get %d: %f", j, ah[j]);
				printf("  Should be : %f\n", a[j]);
				count++;
			}
	}
	printf("Accuracy is %f\n", count/(adims[0]*adims[1]*adims[2]*adims[3]));
}

// This function would print a lot of conv result for testing.
void testConv(float* x, float* conv1, const int* dim1, const int* dim2) {
	int *d_xdims, *d_adims, *d_conv1dims;
	cudaMalloc(&d_xdims, sizeof(int) * 4);
	cudaMalloc(&d_adims, sizeof(int) * 4);
	cudaMalloc(&d_conv1dims, sizeof(int) * 4);
	float *d_x, *d_conv1;
	cudaMalloc(&d_x, sizeof(float) * dim1[0] * dim1[1] * dim1[2] * dim1[3]);
	cudaMalloc(&d_conv1, sizeof(float) * dim2[0] * dim2[1]
						* dim2[2] * dim2[3]);
	cudaMemcpy(d_x, x, sizeof(float) * dim1[0] * dim1[1] * dim1[2] * dim1[3],
						cudaMemcpyHostToDevice);
	cudaMemcpy(d_conv1, conv1, sizeof(float) * dim2[0] * dim2[1]
						* dim2[2] * dim2[3], cudaMemcpyHostToDevice);

  const int adims[] = {dim1[0], (dim1[1] - dim2[0] + 1),
                       (dim1[2] - dim2[1] + 1), dim2[3]};
	cudaMemcpy(d_adims, adims, sizeof(int) * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(d_xdims, dim1, sizeof(int) * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(d_conv1dims, dim2, sizeof(int) * 4, cudaMemcpyHostToDevice);
  auto a = zeros<float>(adims);
  conv_forward_valid(x, dim1, conv1, dim2, a, adims);

	float* ad; //will store the result of first conv
	cudaMalloc(&ad, sizeof(float) * adims[0] * adims[1] * adims[2] * adims[3]);
	cudaMemset(ad, 0, sizeof(float)* adims[0] * adims[1] * adims[2] * adims[3]);
  auto ah = zeros<float>(adims); //host

	dim3 convGrid(adims[0], 1, 1);
	dim3 convBlock(128, 1, 1);
	conv_forward_para1 <<<convGrid, convBlock>>> (d_x, d_xdims, 
				d_conv1, d_conv1dims, ad, d_adims);
	cudaDeviceSynchronize();
	cudaMemcpy(ah, ad, sizeof(float) * adims[0] * adims[1] * adims[2] * adims[3],
				cudaMemcpyDeviceToHost);
	float count = 0.0;
	for (const auto j : range(0, adims[0]*adims[1]*adims[2]*adims[3])) {
			if ((ah[j] - a[j])/a[j] > 1e-2) {
				printf("I get %d: %f", j, ah[j]);
				printf("  Should be : %f\n", a[j]);
				count++;
			}
	}
	printf("Accuracy is %f\n", count/(adims[0]*adims[1]*adims[2]*adims[3]));
}

void testPool(float* x, const int* dim) {
  int *d_dim;
  float *d_x, *d_y;
  cudaMalloc(&d_dim, sizeof(int)*4);
  cudaMalloc(&d_x, sizeof(float) * dim[0] * dim[1] * dim[2] * dim[3]);
  cudaMalloc(&d_y, sizeof(float) * dim[0] * dim[1] * dim[2] * dim[3] / 4);
  cudaMemcpy(d_dim, dim, sizeof(int) * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x,  sizeof(float) * dim[0] * dim[1] * dim[2] * dim[3],
          cudaMemcpyHostToDevice);
  dim3 poolBlock(128, 1, 1);
  float batch_size = 10.0; //number of input images each block should handle
  dim3 poolGrid(ceil(dim[2] * dim[3] / 256.0), ceil(dim[0] / batch_size), 1);
  average_pool_para <<<poolGrid, poolBlock>>> (d_x, d_dim, d_y);
  cudaDeviceSynchronize();
  const int dim2[]   = {dim[0], dim[1] / 2, dim[2] / 2,
                       dim[3]};
  auto y = zeros<float>(dim2);
  auto bh = zeros<float>(dim2);
  cudaMemcpy(y, d_y,  sizeof(float) * dim2[0] * dim2[1] * dim2[2] * dim2[3],
          cudaMemcpyDeviceToHost);
//  average_pool(x, dim, 2, bh, dim2);
  float count = 0.0;
  for (const auto i : range(0, dim2[0] * dim2[1] * dim2[2] * dim2[3])) {
    if ((bh[i] - y[i]) > 1e-5) {
      //printf("I get %d %d %d, %d: %f", i/(dim2[1] * dim2[2] * dim2[3]),
      //      (i/(dim2[3] * dim2[2]))%dim2[1], 
      //  i/(dim2[3])%(dim2[1]), y[i]);
      //printf("  Should be : %f\n", bh[i]);
      count++;
    }
  }
  printf("Accuracy is %f\n", count/(dim2[0]*dim2[1]*dim2[2]*dim2[3]));
}

void testRelu(float* x, const int dim[4]) {
  int *d_dim;
  float *d_x;
  cudaMalloc(&d_dim, sizeof(int)*4);
  cudaMalloc(&d_x, sizeof(float) * dim[0] * dim[1] * dim[2] * dim[3]);
  cudaMemcpy(d_dim, dim, sizeof(int) * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x,  sizeof(float) * dim[0] * dim[1] * dim[2] * dim[3],
          cudaMemcpyHostToDevice);
  int grid = ceil(dim[0] * dim[1] * dim[2] * dim[3] / 128.0);
  relu_para <<<grid, 128>>> (d_x, dim[0] * dim[1] * dim[2] * dim[3]);
  const int dim2[]   = {dim[0], dim[1], dim[2],
                       dim[3]};
  auto out = zeros<float>(dim2);
  auto bh = zeros<float>(dim2);
  cudaMemcpy(out, d_x,  sizeof(float) * dim[0] * dim[1] * dim[2] * dim[3],
          cudaMemcpyDeviceToHost);
//  relu4(x, dim);
  float count = 0.0;
  for (const auto i : range(0, dim[0] * dim[1] * dim[2] * dim[3])) {
    if ((x[i] - out[i]) > 1e-5) {
      //printf("I get %d %d %d, %d: %f", i/(dim2[1] * dim2[2] * dim2[3]),
      //      (i/(dim2[3] * dim2[2]))%dim2[1], 
      //  i/(dim2[3])%(dim2[1]), y[i]);
      //printf("  Should be : %f\n", bh[i]);
      count++;
    }
  }
  printf("Accuracy is %f\n", count/(dim[0]*dim[1]*dim[2]*dim[3]));
}
