#include "range.hpp"
#include "utils.hpp"
#include <stdio.h>
#include <stdlib.h>
#define TILE_WIDTH 32
#define TILE_SIZE 16
#define OUT_WIDTH 8
#define POOL_SIZE 2
__constant__ float W_const[5 * 5 * 32];


//#####################Convolution Kernels###############

/*	For this kernel, each block would work on one tile of output y together.
		The z dimension of block is index for output feature.

		The kernel would loop through each input feature of each input image.

		In one iteration, each block would load the filter for one input feature, 
		as well as a tile of that input feature into shared memory. Then, each 
		thread would perform convolution and add the result to the corresponding 
		y element.
*/
__global__ void conv_forward_para(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) {
	__shared__ float filter[5][5];
	__shared__ float x[TILE_WIDTH][TILE_WIDTH];
	int ox = threadIdx.x + blockIdx.x * OUT_WIDTH;
	int oy = threadIdx.y + blockIdx.y * OUT_WIDTH;
	int tz = blockIdx.z; 
	int in;
	float out;
	// this loop would iterate through input features
	for (const auto i : range(0, wdims[2])) {
		//load filter
		__syncthreads();
		if (threadIdx.x < 5 && threadIdx.y < 5) {
			in = threadIdx.y * wdims[1] * wdims[2] * wdims[3] +
						 threadIdx.x * wdims[2] * wdims[3] + i * wdims[3] + tz;
			filter[threadIdx.x][threadIdx.y] = W[in];
		}
		__syncthreads();
		for (const auto j : range(0, ydims[0])) {
			out = 0.0;
			//load tile first
			in = j * xdims[1]*xdims[2]*xdims[3] + 
					oy * xdims[2] * xdims[3] 	+ ox * xdims[3] + i;
			x[threadIdx.x][threadIdx.y] = X[in];
			__syncthreads();
			//now perform convolution, notice that some thread remain idle
			if (threadIdx.x < OUT_WIDTH && threadIdx.y < OUT_WIDTH) {
				for (const auto ci : range(0, wdims[0])) {
					for (const auto cj : range(0, wdims[1])) {
							out += filter[ci][cj] * x[threadIdx.x + ci][threadIdx.y + cj];
					}
				}
			}
			__syncthreads();
			//add back to y
			if (threadIdx.x < OUT_WIDTH && threadIdx.y < OUT_WIDTH) {
				in = ((j * ydims[1] + oy) * ydims[2] + ox) * ydims[3] + tz;
				Y[in] += out;
			}
			__syncthreads();
		}
	}
}

/*
	Version 2 of conv kernel. ONLY works for second layer of conv
	That means some coefficients are hard coded into this kernel, like
	the dimension of input and output. This is aiming for maximal reuse.

	The first part of code is devoted for loading. ThreadIdx is aligned with
	c, the index for input feature and the continuos index in memory.

	The second part is serveral loops for calculating partial convolution. By
	Partial we refer to adding some of the 25 multiplication to the output
	element, instead of calculating the final output element. ThreadIdx is 
	aligned with m, the output feature index, which is the continuous index
	in this time.

	Block would process one row of input feature. So there are totally
	numOfImages * numOfRows blocks.

	The current version of this kernel is optimized. We have shuffled the order
	of loop to reduce atomicAdd as well as index calculation. We also tried
	to unroll the loop. Please forgive the messy of this performence-oriented 
	version.
	
	The pseudo code is just:
		for each input channel:
		for each filter element positioned at p,q:
			add the result of multiplication to output element at w,h
*/

__global__ void conv_forward_para_v2(int start, int H, int C, int K, int M, float *X, float *W, float *Y) 
{
	//BLOCK size is 128

	// **We abandoned filter loading with reason explained in report.
	//__shared__ float filter[128]; //One element from each filter
	__shared__ float x[12][32]; //One row from each input feature
	__shared__ float y[8 * 64]; //Privitization to reduce atomicAdd

	int w_load = threadIdx.x>>5; //width index when loading , /32
	int c_load = threadIdx.x&31; //input feature index, %32
	int i_load, i_out, h_out;
	int temp =threadIdx.x>>6;
	int row = blockIdx.y;

	//load x first
	//Notice that, each row need not be reloaded at all
	i_load = (((blockIdx.x + start) * H + row) * H + w_load) * C + c_load;
	x[w_load][c_load] = X[i_load];
	x[w_load + 4][c_load] = X[i_load + 4 * C];
	x[w_load + 8][c_load] = X[i_load + 8 * C];
	__syncthreads();
	//finished loading

	for (int q =0; q < 5; q++) {
		//Initialize shared y
		h_out = row - q;
		if (h_out >= 0 && h_out < 8) {
			for (int i = threadIdx.x; i < 8 * 64; i+=blockDim.x) {
				y[i] = 0;
			}
			__syncthreads();
			for (int w = 0; w < 8; w++) {
				i_out = w * 64 + (threadIdx.x&63);
				int i_load_base = ((q * K) * C) * M;
				float out = 0.0;
				for (int p = 0; p < 5; p++) {
				// p, q stand for position index on a filter. p is column index
				// In this order, each thread add to the same element.
					w_load = w+p; //reuse this variable

					//looping through input features
					//explicitly unroll the loop to reduce index calculation
					//Also experimented with intrinsics(a*b+c in single operation)
					i_load = i_load_base + threadIdx.x;
					c_load = temp;
					out += fmaf(x[w_load][c_load] , W[i_load],
							 x[w_load][c_load + 2] * W[i_load + 128])
							+ fmaf(x[w_load][c_load + 4], W[i_load + 256],
							 x[w_load][c_load + 6] * W[i_load + 384])
							+ fmaf(x[w_load][c_load + 8], W[i_load + 512],
							 x[w_load][c_load + 10] * W[i_load + 640])
							+ fmaf(x[w_load][c_load + 12], W[i_load + 768],
							 x[w_load][c_load + 14] * W[i_load + 896]) 
							+ fmaf(x[w_load][c_load + 16] , W[i_load + 1024],
							 x[w_load][c_load + 18] * W[i_load + 1152])
							+ fmaf(x[w_load][c_load + 20], W[i_load + 1280],
							 x[w_load][c_load + 22] * W[i_load + 1408])
							+ fmaf(x[w_load][c_load + 24], W[i_load + 1536],
							 x[w_load][c_load + 26] * W[i_load + 1664])
							+ fmaf(x[w_load][c_load + 28], W[i_load + 1792],
							 x[w_load][c_load + 30] * W[i_load + 1920]) ;
					i_load_base += C * M;
				}
				//In this order of loops, the inner loop need not atomicAdd
				atomicAdd(&y[i_out], out);
				__syncthreads();
			}
			i_out = (((start + blockIdx.x) * 8 + h_out) * 8) * 64;
			for (int i = threadIdx.x; i < 8 * 64; i+=blockDim.x) {
				atomicAdd(&(Y[i + i_out]), y[i]);
			}
			__syncthreads();
		}
	}
}

/* 
	version 2 of the first conv layer. This only works for current dimension.
	More careful treatment of memory accesing patern is performed. Also tried 
	constant memory and shared memory to save filter, however, their efficiency 
	is about the same level of relying on L1 cache. Thus in current form
	we use neither method, but loading directly from W. The cache hit rate, as
	it turned out, is good enough to not limiting performance.

	The idea of the kernel is straight forward: load entire input feature, and
	perform direct convolution. The performance boost mainly comes from reusing
	x, the input feature.

	Notice that this final version contains some little tricks to exploit 
	performance. Like using variables to reduce index calculation, and loop
	unroll. Please dont bother to check the indexing. All reads and writes
	are coalesced (continuous in memory address)
*/
__global__ void conv_forward_para1(int H, int C, int K, int M, const float *X,  const float *W, float *Y) 
{
	//__shared__ float filter[5*5*32];
	__shared__ float x[784];
	int tx = threadIdx.x; 
	int in = blockIdx.x * H * H * C;
	int w = threadIdx.x>>5;
	int c = threadIdx.x&31;
	int H_out = H - K + 1;
	float out;
	// load the entire input feature first
	for (int offset = tx; offset < 784; offset += blockDim.x) {
		x[offset] = X[in + offset];
	}
	__syncthreads();
	//for (int offset = tx; offset < 800; offset += blockDim.x) {
	//	filter[offset] = W[offset];
	//}
	__syncthreads();
	int in_base = ((blockIdx.x * H_out ) * H_out + w) * M + c;
	int h_out = blockIdx.y;
		in = in_base + h_out * H_out * M;
		for (int w_out = w; w_out < 24; w_out += 4) {
			out = 0.0f;
			int x_base = h_out*28 + w_out;
			for (int p = 0; p < 5; p++) {
				// wdims[1] * wdims[2] * wdims[3] = 5*1*32=160
				// wdims[2] * wdims[3] = 32
				int w_idx = p * 160 + c;
					out += W_const[w_idx] * x[x_base]
							+ W_const[w_idx + 32] * x[x_base + 1]
							+ W_const[w_idx + 64] * x[x_base + 2]
							+ W_const[w_idx + 96] * x[x_base + 3]
							+ W_const[w_idx + 128] * x[x_base + 4];
				x_base +=  28;
			}
			Y[in] = out;
			in += 4 * M;
		}
}

/*
	A further experiment on the idea of utilizing massive shared memory.
	Almost the same structure as version two of conv1 layer, but requires
	more effort in indexing etc.

	Techniques like loop unrolling are described in report Other Optimization
	section.

	We also integrated the following average pool and relu into this layer.
	No observable effect in performance.
*/

__global__ void conv2_izzy(int block_start, float *INPUT,float *OUTPUT,float *mask)
{
      ///let the block size be 64*16=1024
      __shared__ float input_img[4608];  /// 1 picture
      __shared__ float result_temp[4096]; /// 16KB, 88KB in total, slightly dangerous since pushed things to extreme. 90KB shared memory at max assumed
      int x_pos= threadIdx.x; /// 64
      int y_pos= threadIdx.y; /// 16
      int pos=x_pos+64*y_pos;

      result_temp[pos]=0;
      result_temp[pos+1024]=0;
      result_temp[pos+2048]=0;
      result_temp[pos+3072]=0;
			
      ///loading all the data
      for (int i=0;i<4;i++)
        input_img[i*1024+pos]= INPUT[(block_start + blockIdx.x)*4608+i*1024+pos];
			
      if(pos<512)
        input_img[4096+pos]=INPUT[(block_start + blockIdx.x)*4608+4096+pos];
			float out = 0.0;
			/*
			//For reference, we keep the unoptimized computation part here

			for(int i=0;i<8;i++)
        for(int j=0;j<8;j++)
					for(int k=0;k<5;k++)
						for(int l=0;l<5;l++) {
              atomicAdd(&result_temp[(i*8+j)*64+x_pos],input_img[((i+k)*12+j+l)*32+y_pos]*mask[((k*5+l)<<11)+y_pos*64+x_pos]);
              atomicAdd(&result_temp[(i*8+j)*64+x_pos],input_img[((i+k)*12+j+l)*32+y_pos+16]*mask[((k*5+l)<<11)+(16+y_pos)*64+x_pos]);
						}
			*/

		int r_pos = x_pos;
		int i_pos = y_pos;
		int i_pos2;
		int m_pos = y_pos*64+x_pos;
		for(int i=0;i<8;i++)
      {
        r_pos = (i<<9) + x_pos;
        for(int j=0;j<8;j++)
        {
          i_pos = ((i * 12) <<5) + (j << 5) + y_pos;
          i_pos2 = i_pos + 16;
					out = 0;
          for(int k=0;k<5;k++)
          {
							int m_pos1 = ((k*5)<<11)+m_pos;
							int m_pos2 = m_pos1 + 1024;
							// Unrolled the loop in l
							out += (input_img[i_pos]*mask[m_pos1]
									+ input_img[i_pos2]*mask[m_pos2])
									+ (input_img[i_pos + 32]*mask[m_pos1 + 2048]
									+ input_img[i_pos2 + 32]*mask[m_pos2 + 2048])
									+ (input_img[i_pos + 64]*mask[m_pos1 + 4096]
									+ input_img[i_pos2 + 64]*mask[m_pos2 + 4096])
									+ (input_img[i_pos + 96]*mask[m_pos1 + 6144]
									+ input_img[i_pos2 + 96]*mask[m_pos2 + 6144])
									+ (input_img[i_pos + 128]*mask[m_pos1 + 8192]
									+ input_img[i_pos2 + 128]*mask[m_pos2 + 8192]);
							i_pos += 384;
							i_pos2 += 384;
          }
					atomicAdd(&result_temp[r_pos], out);
          r_pos += 64;
        }
      }
      __syncthreads();

			// Considering that we have the entire output feature in our shared memory
			// We could integrate relu and average pool here
			// Would save some global writes(only quarter elements need to be written
			// But brings no observable speed up
			int out_y = threadIdx.y >> 2;
			int out_x = threadIdx.y & 3;
			out = 0;
			float sum1 = result_temp[(2*out_y * 8 + out_x*2 ) * 64 + x_pos];
			float sum2 = result_temp[(2*out_y * 8 + out_x*2  + 1) * 64 + x_pos];
			float sum3 =  result_temp[((2*out_y + 1) * 8 + 2*out_x ) * 64 + x_pos];
			float sum4 =  result_temp[((2*out_y + 1) * 8 + 2*out_x  + 1) * 64 + x_pos];
			sum1 = (sum1 < 0)? 0 : sum1;
			sum2 = (sum2 < 0)? 0 : sum2;
			sum3 = (sum3 < 0)? 0 : sum3;
			sum4 = (sum4 < 0)? 0 : sum4;
			out = (sum1 + sum2 + sum3 + sum4) * 0.25;

			OUTPUT[(block_start + blockIdx.x) * 16*64 + out_y * 4 * 64 + out_x * 64 + x_pos] = out;
}


/*
	The next two are attempts for stream. Since stream is not that useful after
	our experiment, these two are abandoned.
*/

__global__ void conv_forward_para1_stream2(int blockStart, int H, int C, int K, int M, const float *X,  const float *W, float *Y) 
{
	//__shared__ float filter[5*5*32];
	__shared__ float x[784];
	int tx = threadIdx.x; 
	int in = (blockStart + blockIdx.x) * H * H * C;
	int w = threadIdx.x>>5;
	int c = threadIdx.x&31;
	int H_out = H - K + 1;
	float out;
	// load the entire input feature first
	for (int offset = tx; offset < 784; offset += blockDim.x) {
		x[offset] = X[in + offset];
	}
	__syncthreads();
	//for (int offset = tx; offset < 800; offset += blockDim.x) {
	//	filter[offset] = W[offset];
	//}
	__syncthreads();
	int in_base = (((blockStart+blockIdx.x) * H_out ) * H_out + w) * M + c;
	for (int h_out = 0; h_out < 24; h_out++) {
		in = in_base + h_out * H_out * M;
		for (int w_out = w; w_out < 24; w_out += 4) {
			out = 0.0f;
			int x_base = h_out*28 + w_out;
			for (int p = 0; p < 5; p++) {
				// wdims[1] * wdims[2] * wdims[3] = 5*1*32=160
				// wdims[2] * wdims[3] = 32
				int w_idx = p * 160 + c;
					out += W[w_idx] * x[x_base + 0]
							+ W[w_idx + 32] * x[x_base + 1]
							+ W[w_idx + 64] * x[x_base + 2]
							+ W[w_idx + 96] * x[x_base + 3]
							+ W[w_idx + 128] * x[x_base + 4];
				x_base +=  28;
			}
			Y[in] = out<0 ? 0 : out;
			in += 4 * M;
		}
	}
}

__global__ void conv_forward_para1_stream(int block_start ,const float *X, const int xdims[4], const float *W, const int wdims[4], float *Y, const int ydims[4]) 
{
	/*
		Instead of invoking multiple blocks, this kernel only process one block i.e. one input image. The previous code can be easily inherited by passing parameter blck_idx
	*/
	__shared__ float x[784];
	int tx = threadIdx.x; 
	int in = (block_start+blockIdx.x)*xdims[1]*xdims[2]*xdims[3];
	//int in = blck_idx* xdims[1] * xdims[2] * xdims[3];
	int w = threadIdx.x>>5;
	int c = threadIdx.x&31;
	float out;
	// load the entire input feature first
	for (int offset = tx; offset < 784; offset += blockDim.x) {
		x[offset] = X[in + offset];
	}
	__syncthreads();
	for (int h_out = 0; h_out < 24; h_out++) {
		for (int w_out = w; w_out < 24; w_out += 4) {
			out = 0.0f;
			for (int p = 0; p < 5; p++) {
				for (int q = 0; q < 5; q++) {
					out += W_const[(p * wdims[1] + q) * wdims[2] * wdims[3] + c] * x[(h_out+p)*28 + w_out + q];
				}
			}
			in = (((block_start+blockIdx.x)* ydims[1] + h_out) * ydims[2] + w_out) * ydims[3] + c;
			Y[in] = out<0?0:out;
		}
	}
}

//######################Convolution layers end###########


/*
	Parallelized kernel for pooling. W is the width of input feature; C is num
	of Channels.
*/
__global__ void average_pool_para(int start, int W, int C, float *X, float *Y) 
{
	/*	This is 128 * 4, so that each thread have 4 needed element ready
		in shared memory */
	__shared__ float x[256][2]; 	
	int tx = threadIdx.x;
	int w = tx/C + blockIdx.x * (128/C);
	int c = tx % C;
	int ix, sx, ox;
	float out;
	int i = blockIdx.y + start;
	int tx_temp =tx/C;
	int temp= W>>1;
		for (int row = 0; row < W; row+=2) {
			out = 0.0;
			//each thread would now calculate the average of 2x2 pool
			//memory access is coalesced in this pattern
			
			ix = (i * W + row)* W * C + tx + blockIdx.x * blockDim.x * 2;
			x[tx][0] = X[ix];
			ix = ix + blockDim.x;
			x[tx + blockDim.x][0] = X[ix];
			ix = (i * W + row + 1) * W * C + tx + blockIdx.x * blockDim.x * 2;
			x[tx][1] = X[ix];
			ix = ix + blockDim.x;
			x[tx + blockDim.x][1] = X[ix];
			__syncthreads();
			sx = (tx_temp) * C * 2 + c;
			out = (x[sx][0] + x[sx][1] + x[sx + C][0] + x[sx + C][1])*0.25; // = *0.25
			ox = i * (temp) * (temp) * C + (row >> 1) * (temp) * C	+ w * C + c;
			Y[ox] = out;
			__syncthreads();
	}
}

// A simple kernel where each thread map one element of X
__global__ void relu_para(float *X, const int len) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < len) {
		X[i] = (X[i] < 0) ? 0 : X[i]; 
	}
}

/*
	Fully forward kernel
	Almost identical to MP matrix multiplication. Utilized tiling to reuse
	data.
*/

__global__ void fully_forward_p(int block_start, int xd0, int xd1, int wd0, int wd1, float *X, float *W, float *Y)
{
      ///[0]= row, [1]= col, X*W=Y
      /// let a 2-d grid perform the product
      // TILE_SIZE = 32
      int x_row=xd0 + block_start, x_col=xd1;
      int w_row=wd0, w_col=wd1;

      int block_x=blockIdx.x;
      int block_y=blockIdx.y;
      int tile_x = threadIdx.x;
      int tile_y =threadIdx.y;
      int index_row = block_y*TILE_SIZE+tile_y + block_start;
      int index_col= block_x*TILE_SIZE+tile_x;

      int phase = (x_col%TILE_SIZE)==0? (x_col/TILE_SIZE):(x_col/TILE_SIZE+1);
      float sum=0;
      __shared__ float tileW[TILE_SIZE][TILE_SIZE];//the tile for W matrix
      __shared__ float tileX[TILE_SIZE][TILE_SIZE];//the tile for X matrix

      for (int i=0; i<phase;i++)
      {
            /// how to take advantage of coalescing
            if (index_col< w_col && i*TILE_SIZE+tile_y< w_row)
                  tileW[tile_y][tile_x] = W[(i*TILE_SIZE+tile_y)*w_col+index_col];
            else
                  tileW[tile_y][tile_x]=0;

            if(index_row<x_row && i*TILE_SIZE+tile_x < x_col)
                  tileX[tile_y][tile_x] = X[x_col*index_row+i*TILE_SIZE+tile_x];
            else
                  tileX[tile_y][tile_x]=0;

            __syncthreads();
						// theoretically the next line can help reducing index calculation
						// sadly no observable effect since this kernel takes little time
						#pragma unroll
            for (int j=0; j<TILE_SIZE;j++)
                  sum +=tileX[tile_y][j]*tileW[j][tile_x];
            __syncthreads();
      }

      if (index_col< w_col && index_row < x_row) {
            Y[index_row*w_col+index_col] = sum;
			}

}

/*
	Another fully_forward kernel. Much simpler implementation. Only loads
	one row of X into shared memory. Different blocks would take care different
	rows. Since Y has 128 columns, out 128 threads can write one row of Y at 
	one time. Memory read and write are in continuous memory address. Rely on 
	L1 cache for filter.

	The speed of this kernel is almost identical with the above tiled
	multiplication one in our project.
*/

__global__ void fully_forward_p1(int block_start, int xd0, int xd1, int wd0, int wd1, float *X, float *W, float *Y)
{
      ///suitable block & grid size?
      ///[0]= row, [1]= col, X*W=Y
      /// let a 2-d grid perform the product
      // TILE_SIZE = 32
			__shared__ float x[1024];
			int tx = threadIdx.x;
			int row = blockIdx.x + block_start;
			int offset = row * 1024 + threadIdx.x;
			for (int i = 0; i < 8; i++) {
				x[threadIdx.x + i * 128] = X[offset + i * 128];
			}
			__syncthreads();
			
			float out = 0.0;
			#pragma unroll
			for(int i = 0; i < 1024; i++) {
				out += x[i] * W[i * 128 + tx];
			}

      Y[row * 128 + tx] = (out < 0) ? 0 : out;

}

// An straight forward implementation of last layer.
__global__ void argmax_p(const float *X, const int tot, const int len, int*Y) {
	int max_idx = 0;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < tot) {
	float max = X[tid * len];
	for (int j = 0; j < len; j++) {
		float elem = X[(tid * len) + j];
		if (elem > max) {
			max_idx = j;
			max     = elem;
		}
	}
		Y[tid] = max_idx;
	}
}
//##################Below are for matrix unroll technique


// each thread builds K*K (1 conv region of 1 feat map)
__global__ void unroll_kernel(int C, int H, int W, int K, float* X, float* X_unroll){
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y;
	int H_o = H - K + 1;
	int W_o = W - K + 1;
	int W_unroll = C*K*K;	// width of unrolled feature maps
	if(t < C*H_o*W_o){
		int c = t % C;	// which input feat map we're looking at
		int s = t / C;
		int unroll_row_idx = s;

		int in_row = s >> 3;
		int in_col = s & 7;
		int idx_base = n*H*W*C + in_row*W*C + in_col*C + c;
		int unroll_idx_base = n*H_o*W_o*W_unroll + unroll_row_idx*W_unroll + c*K*K;
		for(int p = 0; p < K; p++){
				int unroll_idx = p*K + unroll_idx_base;
				int idx = idx_base + p*W*C;
			for(int q = 0; q < K; q++){
				// iterate right through the section column
				X_unroll[unroll_idx] = X[idx];
				unroll_idx += 1;
				idx += C;
			}
		}
	}	
}

__global__ void GEMM_kernel(float* A, float* B, float* C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns){
	__shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
  
	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x;  int ty = threadIdx.y;

	int Row = by*TILE_WIDTH + ty;
	int Col = bx*TILE_WIDTH + tx;
	int n = blockIdx.z;  
  
	float Pvalue = 0.0;
  
	for(int m = 0; m < (numAColumns - 1)/TILE_WIDTH + 1; m++){
		// load sub-tiles
		if(Row < numARows && (m*TILE_WIDTH + tx) < numAColumns) 
		  subTileA[ty][tx] = A[n*numARows*numAColumns + Row*numAColumns + m*TILE_WIDTH + tx];
		else
		  subTileA[ty][tx] = 0.0;
		if(Col < numBColumns && (m*TILE_WIDTH + ty) < numBRows) 
		  subTileB[ty][tx] = B[(m*TILE_WIDTH + ty)*numBColumns + Col];
		else
		  subTileB[ty][tx] = 0.0;
		__syncthreads();
		for(int k = 0; k < TILE_WIDTH; k+=16) {
		  Pvalue += subTileA[ty][k]*subTileB[k][tx]
		  				+ subTileA[ty][k+1]*subTileB[k+1][tx]
		  				+ subTileA[ty][k+2]*subTileB[k+2][tx]
		  				+ subTileA[ty][k+3]*subTileB[k+3][tx]
		  				+	subTileA[ty][k+4]*subTileB[k+4][tx]
		  				+ subTileA[ty][k+5]*subTileB[k+5][tx]
		  				+ subTileA[ty][k+6]*subTileB[k+6][tx]
		  				+ subTileA[ty][k+7]*subTileB[k+7][tx]
		  				+ subTileA[ty][k+8]*subTileB[k+8][tx]
		  				+ subTileA[ty][k+9]*subTileB[k+9][tx]
		  				+ subTileA[ty][k+10]*subTileB[k+10][tx]
		  				+	subTileA[ty][k+11]*subTileB[k+11][tx]
		  				+ subTileA[ty][k+12]*subTileB[k+12][tx]
		  				+ subTileA[ty][k+13]*subTileB[k+13][tx]
		  				+ subTileA[ty][k+14]*subTileB[k+14][tx]
		  				+ subTileA[ty][k+15]*subTileB[k+15][tx];
		}
		__syncthreads();
	}
	if(Row < numCRows && Col < numCColumns)
		C[n*numCRows*numCColumns + Row*numCColumns + Col] = Pvalue; 		
}

__global__ void GEMM_kernel_v2(float* A, float* B, float* C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
      ///suitable block & grid size?
      ///[0]= row, [1]= col, X*W=Y
      /// let a 2-d grid perform the product
      // TILE_SIZE = 32
			__shared__ float x[800];
			int tx = threadIdx.x;
			int row = blockIdx.x;
			int w_x = tx & 63;
			int w_y = tx / 64;
			int offset = blockIdx.y * 64 * 800 + row * 800;
			for (int load_idx = tx; load_idx < 800; load_idx+=blockDim.x) {
				x[load_idx] = A[offset + load_idx];
			}
			__syncthreads();
			
			float out = 0.0;
			#pragma unroll
			for(int i = w_y; i < 800; i+=16) {
				out += x[i] * B[i * 64 + w_x]
						 + x[i + 2] * B[i * 64 + 128 + w_x]
						 + x[i + 4] * B[i * 64 + 256 + w_x]
						 + x[i + 6] * B[i * 64 + 384 + w_x]
						 + x[i + 8] * B[i * 64 + 512 + w_x]
						 + x[i + 10] * B[i * 64 + 640 + w_x]
						 + x[i + 12] * B[i * 64 + 768 + w_x]
						 + x[i + 14] * B[i * 64 + 896 + w_x];
			}
			__syncthreads();
      atomicAdd(&C[blockIdx.y * 64 * 64 + row * 64 + w_x], out);

}
