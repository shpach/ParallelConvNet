#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>

#include <hdf5.h>

#include "kernels.cu"
#include "serial.cu"
#include "query.hpp"
#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10
#define TILE_SIZE 16
///1 activate multistream version
#define ASYNC_SIGN 0

// 1 to activiate unrolled version
#define UNROLL_SIGN 0

static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32};
static int conv2dims[] = {5, 5, 32, 64};
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};

__host__ static int loadData(float *x, float *y) {
  // Open the data file
  const auto file_id =
      H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset x and y
  const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
  const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

  // Get the dataset x dimensions
  const auto xspace = H5Dget_space(x_id);
  const auto xndims = H5Sget_simple_extent_ndims(xspace);
  assert(xndims == 4);

  hsize_t input_dims[xndims];
  H5Sget_simple_extent_dims(xspace, input_dims, NULL);
  if (input_dims[0] != FLAGS_batch_size) {
    std::cout << "data size does not match batch size specified!\n";
    return 1; // return error
  }
  std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1]
            << " x " << input_dims[2] << " x " << input_dims[3] << "\n";

  // Read the dataset x and y
  check_success(
      H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
  check_success(
      H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

  // Close the dataset x and y
  check_success(H5Dclose(x_id));
  check_success(H5Dclose(y_id));

  // Close the file
  check_success(H5Fclose(file_id));

  // return success
  return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
  // Open the model file
  const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset
  const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
  const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
  const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
  const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

  // Read the dataset
  check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv1));
  check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv2));
  check_success(
      H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
  check_success(
      H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

  // Close the dataset x and y
  check_success(H5Dclose(conv1_id));
  check_success(H5Dclose(conv2_id));
  check_success(H5Dclose(fc1_id));
  check_success(H5Dclose(fc2_id));

  // Close the file
  check_success(H5Fclose(file_id));
}

// CPU function to unroll weights
void unrollWeights(const float* W, const int wdims[4], float* W_unroll){
	int unroll_idx = 0;
	int C = wdims[2];
	int filter_H = wdims[0];
	int filter_W = wdims[1];
	int M = wdims[3];	

	for(int c = 0; c < C; c++){
		for(int h = 0; h < filter_H; h++){
			for(int w = 0; w < filter_W; w++){
				for(int m = 0; m < M; m++){
					int idx = h*filter_W*C*M + w*C*M + c*M + m;
					W_unroll[unroll_idx] = W[idx];
					unroll_idx++;
				}
			}
		}
	} 
}

// Used to launch matrix unroll version of convolution kernel
void convLayer_forward_sgemm(float *d_X, const int xdims[4], 
							const float* W, const int wdims[4],
							float *d_Y, const int ydims[4]){

	int N = xdims[0];
	int M = ydims[3];
	int C = xdims[3];
	int H_i = xdims[1];
	int W_i = xdims[2];
	int K = wdims[0];
	
	int H_o = ydims[1];
	int W_o = ydims[2];	

	// dimensions for GEMM
	int height_X_unroll = H_o*W_o;
	int width_X_unroll = C*K*K;

	int height_W_unroll = C*K*K;	
	int width_W_unroll = M;
	printf("hx %d, wx %d, ww%d\n", height_X_unroll, width_X_unroll, width_W_unroll);


	float* W_unroll = (float*) malloc(sizeof(float)*height_W_unroll*width_W_unroll);

	const auto weight_start = now();	
 	unrollWeights(W, wdims, W_unroll);
	const auto weight_end = now();

	const auto elapsed_weights =
      std::chrono::duration<double, std::milli>(weight_end - weight_start).count();

	std::cout << "Unroll Weights Time:\t" << elapsed_weights << std::endl; 


	const auto conv_start = now();	


	/* --- BEGIN: Unrolling X --- */

	/* --- BEGIN: Memory Allocations --- */
	float *d_X_unroll;

	int width_unroll = C*K*K;
	int height_unroll = H_o*W_o;
	int sizeUnroll = sizeof(float)*width_unroll*height_unroll;

	cudaMalloc(&d_X_unroll, N * sizeUnroll);
	/* --- END: Memory Allocations --- */


	const auto unroll_start = now();
	/* --- BEGIN: Launch Kernel --- */
	dim3 dimBlockUnroll(1024, 1, 1);
	dim3 dimGridUnroll(ceil((C*H_o*W_o)/1024.0), N, 1);
	unroll_kernel<<<dimGridUnroll, dimBlockUnroll>>>(C, H_i, W_i, K, d_X, d_X_unroll);
	/* --- END: Launch Kernel --- */
	cudaDeviceSynchronize();
	const auto unroll_end = now();
	const auto elapsed_unroll =
	std::chrono::duration<double, std::milli>(unroll_end - unroll_start).count();
	std::cout << "Unroll_X Time:\t" << elapsed_unroll << std::endl;
	/* --- END: Unrolling X --- */


	/* --- BEGIN: Matrix Multiplication (X_unroll to W_unroll) --- */
	const auto gemm_start = now();	

	float *d_W;

	/* --- BEGIN: Memory Allocations --- */
	int sizeW = sizeof(float) * height_W_unroll * width_W_unroll;
	cudaMalloc(&d_W, sizeW);
	/* --- END: Memory Allocations --- */

	cudaMemcpy(d_W, W_unroll, sizeW, cudaMemcpyHostToDevice);


	/* --- BEGIN: Kernel Launch --- */
	dim3 dimBlockGEMM(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 dimGridGEMM(ceil(width_W_unroll/(TILE_WIDTH * 1.0)), ceil(height_X_unroll/(TILE_WIDTH * 1.0)), N); 
	//GEMM_kernel<<<dimGridGEMM, dimBlockGEMM>>>(d_X_unroll, d_W, d_Y, height_X_unroll, width_X_unroll, height_W_unroll, width_W_unroll, height_X_unroll, width_W_unroll);
	cudaFuncSetCacheConfig(GEMM_kernel_v2, cudaFuncCachePreferL1);
	dim3 block(128, 1, 1);
	dim3 grid(64, N, 1);
	GEMM_kernel_v2<<<grid, block>>>(d_X_unroll, d_W, d_Y, height_X_unroll, width_X_unroll, height_W_unroll, width_W_unroll, height_X_unroll, width_W_unroll);	
	/* --- END: Kernel Launch --- */


	/* --- BEGIN: Free Device Memory --- */
	cudaFree(d_W);
	/* --- END: Free Device Memory --- */



	const auto gemm_end = now();
	const auto elapsed_gemm =	
	std::chrono::duration<double, std::milli>(gemm_end - gemm_start).count();
	std::cout << "GEMM Time:\t" << elapsed_gemm << std::endl;
	/* --- END: Matrix Multiplication (X_unroll to W_unroll) --- */

	

	const auto conv_end = now();
	const auto elapsed_conv =
      std::chrono::duration<double, std::milli>(conv_end - conv_start).count();

	std::cout << "Conv Time:\t" << elapsed_conv << std::endl;	
	
}

// wrap up of all operations. Including gpu memory allocation and kernel launch
void forward_operation_parallel(float *x, float *conv1, float *conv2, float *fc1, float *fc2, int *out) {
	float *d_x, *d_conv1, *d_conv2, *d_fc1, *d_fc2;
	int* d_out;
	float *a, *b, *c, *d, *e, *f; //will store the result after each layer
	
	cudaMalloc(&d_x, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3]);
	cudaMemcpyToSymbol(W_const, conv1, 5*5*32*sizeof(float));

	//define all the dimensions
	int pool_size = 2;
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
	
  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                       adims[3]};
  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),(bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
                       cdims[3]};
  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};
  const int edims[] = {ddims[0], fc1dims[1]};
  const int fdims[] = {edims[0], fc2dims[1]};
	
	cudaMalloc(&d_conv1, sizeof(float)*conv1dims[0]*conv1dims[1]*conv1dims[2]*conv1dims[3]);
	cudaMalloc(&d_conv2, sizeof(float)*conv2dims[0]*conv2dims[1]*conv2dims[2]*conv2dims[3]);
	cudaMalloc(&d_fc1, sizeof(float) * fc1dims[0] * fc1dims[1]);
	cudaMalloc(&d_fc2, sizeof(float) * fc2dims[0] * fc2dims[1]);
	cudaMalloc(&a, sizeof(float) * adims[0] * adims[1] * adims[2] * adims[3]);
	cudaMalloc(&b, sizeof(float) * bdims[0] * bdims[1] * bdims[2] * bdims[3]);
	cudaMalloc(&c, sizeof(float) * cdims[0] * cdims[1] * cdims[2] * cdims[3]);
	cudaMalloc(&d, sizeof(float) * ddims[0] * ddims[1] * ddims[2] * ddims[3]);
  cudaMalloc(&e, sizeof(float) * edims[0] * edims[1]);
	cudaMalloc(&f, sizeof(float) * fdims[0] * fdims[1]);
	cudaMalloc(&d_out, sizeof(int) * fdims[0]);

	cudaMemcpy(d_conv1, conv1, sizeof(float) * conv1dims[0] * conv1dims[1]* conv1dims[2] * conv1dims[3], cudaMemcpyHostToDevice);
	cudaMemcpy(d_conv2, conv2, sizeof(float) * conv2dims[0] * conv2dims[1]* conv2dims[2] * conv2dims[3], cudaMemcpyHostToDevice);
	cudaMemcpy(d_fc1, fc1, sizeof(float) * fc1dims[0] * fc1dims[1], cudaMemcpyHostToDevice);
	cudaMemcpy(d_fc2, fc2, sizeof(float) * fc2dims[0] * fc2dims[1], cudaMemcpyHostToDevice);
	cudaMemset(a, 0, sizeof(float) * adims[0] * adims[1] * adims[2] * adims[3]);
	cudaMemset(c, 0, sizeof(float) * cdims[0] * cdims[1] * cdims[2] * cdims[3]);
	
	#if !ASYNC_SIGN 	
	// First move all related data to device
		cudaMemcpy(d_x, x, sizeof(float) * xdims[0] * xdims[1] * xdims[2] * xdims[3],cudaMemcpyHostToDevice);
	#endif
	

	
	// below is our attemp to use stream. Can ignore because they are not useful
	#if ASYNC_SIGN 
	{	
		// num of stream used
		// the set of images is divided evenly between streams
		// for example, 10000 images with 10 streams would have each stream
		// working on a batch of 1000 images
		int MAX_STREAM_KPL = 5;
		int batch_size = adims[0] / MAX_STREAM_KPL;
		int len = xdims[1]*xdims[2]*xdims[3];
		/// dimension of grid is 16
		cudaStream_t stream_zoo[MAX_STREAM_KPL];
		for(int i=0;i<MAX_STREAM_KPL;i++) {
			cudaStreamCreate(&stream_zoo[i]);
		}
		for(int i=0;i< MAX_STREAM_KPL;i++) {
			cudaMemcpyAsync(&d_x[i*batch_size*len],&x[i*batch_size*len],sizeof(float)*len*batch_size,cudaMemcpyHostToDevice,stream_zoo[i]);
		}

		for(int i=0;i< MAX_STREAM_KPL;i++) {
			conv_forward_para1_stream2<<<batch_size,128,0,stream_zoo[i]>>>(batch_size*i,xdims[1], xdims[3], conv1dims[1], conv1dims[3], d_x, d_conv1,a);
		}
		dim3 poolBlock(128, 1, 1);
		dim3 poolGrid(ceil(adims[2] * adims[3] / 256.0), batch_size, 1);
		for(int i=0;i< MAX_STREAM_KPL;i++) {
  		average_pool_para <<<poolGrid, poolBlock, 0, stream_zoo[i]>>> (batch_size*i, adims[2], adims[3], a, b);
		}

		dim3 grid (batch_size, 1,1);
		dim3 block (64, 16, 1);
		for(int i=0;i< MAX_STREAM_KPL;i++) {
			conv2_izzy <<<grid, block, 0, stream_zoo[i]>>> (batch_size*i, b, d, d_conv2);
		}
			
		int blck_dim_y,blck_dim_x;
		blck_dim_y = batch_size%TILE_SIZE==0?batch_size/TILE_SIZE:batch_size/TILE_SIZE+1;
		blck_dim_x = edims[1]%TILE_SIZE==0?edims[1]/TILE_SIZE:edims[1]/TILE_SIZE+1;
		dim3 grd_dim(blck_dim_x,blck_dim_y,1);
		dim3 blk_dim(TILE_SIZE,TILE_SIZE,1);
		for(int i=0;i< MAX_STREAM_KPL;i++) {
		fully_forward_p1<<<grd_dim, blk_dim, 0, stream_zoo[i]>>>(batch_size*i, batch_size, ddims2[1], fc1dims[0], fc1dims[1], d, d_fc1, e);
	 	}

		blck_dim_x = fdims[1]%TILE_SIZE==0?fdims[1]/TILE_SIZE:fdims[1]/TILE_SIZE+1;
		dim3 grd_dim2(blck_dim_x,blck_dim_y,1);
		for(int i=0;i< MAX_STREAM_KPL;i++) {
		fully_forward_p<<<grd_dim2, blk_dim, 0, stream_zoo[i]>>>(batch_size*i, batch_size, edims[1], fc2dims[0], fc2dims[1], e,d_fc2,f);
		}


		cudaStreamSynchronize(0);
		for(int i=0;i<MAX_STREAM_KPL;i++)
			cudaStreamDestroy(stream_zoo[i]);
	}
	#else
	// Normal part of code
	{
	// First layer of convolution
		{
		dim3 convGrid(adims[0], 24, 1);
		dim3 convBlock(128, 1, 1);
		conv_forward_para1 <<<convGrid, convBlock>>> (xdims[1], xdims[3], conv1dims[1], conv1dims[3], d_x, d_conv1, a);
		}
	
	// relu
	int reluGrid = ceil(adims[0] * adims[1] * adims[2] * adims[3] / 128.0);
	int reluBlock=128;
  relu_para <<<reluGrid, reluBlock>>> (a, adims[0] * adims[1] * adims[2] * adims[3]);
	
  // average pooling
	dim3 poolBlock(128, 1, 1);
	dim3 poolGrid(ceil(adims[2] * adims[3] / 256.0), adims[0], 1);
  average_pool_para <<<poolGrid, poolBlock>>> (0, adims[2], adims[3], a, b);

	//Another conv layer
	#if UNROLL_SIGN
	// if UNROLL_SIGN is set to one, would run matrix unroll version of conv2 layer
	{
		convLayer_forward_sgemm(b, bdims, conv2, conv2dims, c, cdims);

		reluGrid = ceil(cdims[0] * cdims[1] * cdims[2] * cdims[3] / 128.0);
  	relu_para <<<reluGrid, reluBlock>>> (c, cdims[0] * cdims[1] * cdims[2] * cdims[3]);
		
		dim3 poolGrid2(ceil(cdims[2] * cdims[3] / 256.0), cdims[0], 1);
		average_pool_para <<<poolGrid2, poolBlock>>> (0, cdims[2], cdims[3], c, d);
	}
	#else
	// the version that maximally utilize shared memory
	{
		dim3 grid (cdims[0], 1,1);
		dim3 block (64, 16, 1);
		conv2_izzy <<<grid, block>>> (0, b, d, d_conv2);
	}
	#endif

  // matrix multiplication
	int blck_dim_y,blck_dim_x;
  blck_dim_y = edims[0]%TILE_SIZE==0?edims[0]/TILE_SIZE:edims[0]/TILE_SIZE+1;
  blck_dim_x = edims[1]%TILE_SIZE==0?edims[1]/TILE_SIZE:edims[1]/TILE_SIZE+1;
  dim3 grd_dim(blck_dim_x,blck_dim_y,1);
  dim3 blk_dim(TILE_SIZE,TILE_SIZE,1);
  fully_forward_p1<<<ddims[0], 128>>>(0, ddims2[0], ddims2[1], fc1dims[0], fc1dims[1], d, d_fc1, e);
	
  // matrix multiplication
	blck_dim_y = fdims[0]%TILE_SIZE==0?fdims[0]/TILE_SIZE:fdims[0]/TILE_SIZE+1;
	blck_dim_x = fdims[1]%TILE_SIZE==0?fdims[1]/TILE_SIZE:fdims[1]/TILE_SIZE+1;
	dim3 grd_dim2(blck_dim_x,blck_dim_y,1);
	dim3 blk_dim2(TILE_SIZE,TILE_SIZE,1);
	fully_forward_p<<<grd_dim2, blk_dim2>>>(0, edims[0], edims[1], fc2dims[0], fc2dims[1], e,d_fc2,f);
	}
	#endif


  argmax_p <<<ceil(fdims[0]/128.0), 128>>> (f, fdims[0], fdims[1], d_out);
	cudaMemcpy(out,d_out,sizeof(int)*fdims[0],cudaMemcpyDeviceToHost);

	cudaFree(d_x);
	cudaFree(d_conv1);
	cudaFree(d_conv2);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(d);
	cudaFree(e);
	cudaFree(f);
	cudaFree(d_fc1);
	cudaFree(d_fc2);
}


int main(int argc, char **argv) {

  if (argc != 3 && argc != 4) {
    std::cerr << "\n"
              << "This program performs the forward opertion step for "
                 "Convolutional Neural Network(CNN).  "
                 "Sample usage: \n"
              << argv[0]
              << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
    return -1;
  }
  FLAGS_testdata = std::string(argv[1]);
  FLAGS_model    = std::string(argv[2]);
  if (argc == 3) {
    const std::map<std::string, int> default_batch_sizes{
        {"../data/test2.hdf5", 2},
        {"../data/test10.hdf5", 10},
        {"../data/test100.hdf5", 100},
        {"../data/testfull.hdf5", 10000}};
    const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
    if (batch_size_in_map == default_batch_sizes.end()) {
      std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
      return -1;
    }
    FLAGS_batch_size = batch_size_in_map->second;
  } else if (argc == 4) {
    FLAGS_batch_size = atoi(argv[3]);
  }
  xdims[0] = FLAGS_batch_size;
  rdims[0] = FLAGS_batch_size;

  // Load data into x and y
  float *x = allocate<float>(xdims);
  float *y = allocate<float>(rdims);
  loadData(x, y);

  // Load model
  float *conv1 = allocate<float>(conv1dims);
  float *conv2 = allocate<float>(conv2dims);
  float *fc1   = allocate<float>(fc1dims);
  float *fc2   = allocate<float>(fc2dims);
  loadModel(conv1, conv2, fc1, fc2);

  // Perform foward opertion
  int *out = zeros<int>(FLAGS_batch_size);
	
  // get start time
  const auto start = now();
	
	//Calling the parallel version(will launch kernel inside
  forward_operation_parallel(x, conv1, conv2, fc1, fc2, out);
  // get end time
  const auto end = now();

  // get elapsed time in milliseconds
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Get reference
  int *ref = zeros<int>(FLAGS_batch_size);
  argmax(y, rdims, ref);

  // Calculate correctness
  int num_correct = 0;
  for (const auto i : range(0, FLAGS_batch_size)) {
    if (out[i] == ref[i]) {
      num_correct++;
    }
  }
  std::cout << "Done with " << FLAGS_batch_size << " queries in "
            << "elapsed = " << elapsed << " milliseconds. Correctness: "
            << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

  delete[] x;
  delete[] y;
  delete[] conv1;
  delete[] conv2;
  delete[] fc1;
  delete[] fc2;
  delete[] out;
  delete[] ref;

  return 0;
}
