#include "../../../include/cuda/kernels/boids_kernels.cuh"

#include "../../../include/cuda/helper_math.h"

#include <stdio.h>

using namespace BoidsKernels;

#define PI 3.14159265358979323846264f

__device__ const int blockSize = 256;
// __device__ const float f1 = 315.0f/(64.0f*PI);
// __device__ const float f2 = 45.0f/PI;
// __device__ const float nu = 0.05f;// 3.5 * 0.0075f;


__device__ int3 BoidsKernels::calcGridPosition(float4 pos, int3 grid, float3 gridSize)
{
	int3 gridPosition;
	/*gridPosition.x = __float2int_rd(pos.x * grid.x / gridSize.x);
	gridPosition.y = __float2int_rd(pos.y * grid.y / gridSize.y);
	gridPosition.z = __float2int_rd(pos.z * grid.z / gridSize.z);*/
	gridPosition.x = floor(pos.x * grid.x / gridSize.x);
	gridPosition.y = floor(pos.y * grid.y / gridSize.y);
	gridPosition.z = floor(pos.z * grid.z / gridSize.z);

	return gridPosition;
}


// what index the given gridPos corresponds to in the 1D array of cells
__device__ int BoidsKernels::calcCellIndex(int3 gridPos, int3 grid)
{
	return grid.y * grid.x * gridPos.z + grid.x * gridPos.y + gridPos.x;
}


// calculate spatial has for infinite domains
__device__ int BoidsKernels::calcGridHash(int3 gridPos, int numBuckets)
{
	const uint p1 = 73856093;
	const uint p2 = 19349663;
	const uint p3 = 83492791;
	int n = p1*gridPos.x ^ p2*gridPos.y ^ p3*gridPos.z;
	n %= numBuckets;
	return n;
}


__global__ void BoidsKernels::build_spatial_grid
(
	float4 *pos,
	int *boidsIndex,
	int *cellIndex,
	int numBoids,
	int3 grid,
	float3 gridSize
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	while (index + offset < numBoids){
		// find which grid cell the particle is in
		int3 gridPos = calcGridPosition(pos[index + offset], grid, gridSize);

		// compute cell index 
		int cindex = calcCellIndex(gridPos, grid);

		boidsIndex[index + offset] = index + offset;
		cellIndex[index + offset] = cindex;

		offset += blockDim.x*gridDim.x;
	}
}

__global__ void BoidsKernels::reorder_boids
(
	float4 *pos,
	float4 *spos,
	float4 *vel,
	float4 *svel,
	int *cellStartIndex,
	int *cellEndIndex,
	int *cellIndex,
	int *boidsIndex,
	int numBoids
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	__shared__ int sharedcellIndex[blockSize + 1];  //blockSize + 1

	while (index + offset < numBoids){
		sharedcellIndex[threadIdx.x] = cellIndex[index + offset];
		if (threadIdx.x == blockDim.x - 1)
		{
			if (index + offset + 1 < numBoids){
				sharedcellIndex[threadIdx.x + 1] = cellIndex[index + offset + 1];
			}
			else{
				sharedcellIndex[threadIdx.x + 1] = -1;
			}
		}

		__syncthreads();

		if (sharedcellIndex[threadIdx.x] != sharedcellIndex[threadIdx.x + 1]){
			cellStartIndex[sharedcellIndex[threadIdx.x + 1]] = index + offset + 1;
			// cellEndIndex[sharedcellIndex[threadIdx.x]] = index + offset;
			cellEndIndex[sharedcellIndex[threadIdx.x]] = index + offset + 1;
		}

		// reorder position and velocity
		int p = boidsIndex[index + offset];
		spos[index + offset] = pos[p];
		svel[index + offset] = vel[p];

		offset += blockDim.x*gridDim.x;
	}

	__syncthreads();

	if (threadIdx.x == 0 && blockIdx.x == 0){
		cellStartIndex[sharedcellIndex[0]] = 0;
	}
}


__global__ void BoidsKernels::calculate_boids_direction
(
	float4 *pos,
	float4 *vel,
	float4 *scratch,
	int *cellStartIndex,
	int *cellEndIndex,
	int *cellIndex,
	int *boidsIndex,
	int numBoids,
	int3 grid
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;
	int maxIndex = grid.x*grid.y*grid.z;

	while (index + offset < numBoids){

		float4 position = pos[index + offset];
		float4 velocity = vel[index + offset];

		// 'centre of mass' of local boid positions
		float4 com = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		// 'centre of mass' of local boid velocities
		float4 vcom = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		// number of local boids
		int numLocalBoids = 0;

		// vector to steer boids that are too close to each other away
		float4 close = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		int c = cellIndex[index + offset];
		for (int k = -1; k <= 1; k++){
			for (int j = -1; j <= 1; j++){
				for (int i = -1; i <= 1; i++){
					int hash = c + grid.y*grid.x*k + grid.x*j + i;
					if (hash >= 0 && hash < maxIndex){

						// loop through all boids in this cell
						for (int l = cellStartIndex[hash]; l < cellEndIndex[hash]; l++){
							float4 p = pos[l];
							float4 v = vel[l];

							com += p;
							com += v;
							numLocalBoids++;

							float4 r = position - p;
							float radius2 = r.x*r.x + r.y*r.y + r.z*r.z;
							if (radius2 < 0.1f){
								close = close + r;
							}
						}
					}
				}
			}
		}

		com = com / numLocalBoids;
		vcom = vcom / numLocalBoids;

		scratch[index + offset] += 0.01f * (com - position) + 0.125f * (vcom - velocity) + close;

		offset += blockDim.x * gridDim.x;
	}
}

__global__ void BoidsKernels::update_boids
(
	float4 *pos,
	float4 *vel,
	float4 *scratch,
	float *model,
	float dt,
	float h,
	int numBoids,
	float3 gridSize
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	while (index + offset < numBoids){
		float4 my_pos = pos[index + offset];
		float4 my_vel = vel[index + offset];
		float4 s = scratch[index + offset];

		my_vel = my_vel + s;
		my_pos = my_pos + my_vel;


		//pos[index + offset] = my_pos;
		//vel[index + offset] = my_vel;

		// use column major ordering in line with glm and opengl
		model[16 * (index + offset) + 12] = my_pos.x; 
		model[16 * (index + offset) + 13] = my_pos.y;
		model[16 * (index + offset) + 14] = my_pos.z;
		model[16 * (index + offset) + 15] = 1.0f;

		offset += blockDim.x * gridDim.x;
	}
}



// // penalty force boundary conditions http://n-e-r-v-o-u-s.com/education/simulation/week6.php
// __device__ void FluidKernels::boundary(float4 *pos, float4 *vel, float h, float3 gridSize)
// {
// 	float d;
// 	float n = 1.0f;

// 	d = -n*(gridSize.x - (*pos).x) + h;
// 	if(d > 0.0f){
// 		(*pos).x += d*(-0.5f);
// 		(*vel).x -= 1.9*(*vel).x*(-0.5f)*(-0.5f);

// 		// (*acc).x += 10000*(-n)*d;
// 		// (*acc).x += -0.9 * (*vel).x*(-n)*(-n);
// 	} 

// 	d = n*(0.0f - (*pos).x) + h;
// 	if(d > 0.0f){
// 		(*pos).x += d*(0.5f);
// 		(*vel).x -= 1.9*(*vel).x*(0.5f)*(0.5f);

// 		// (*acc).x += 10000*(n)*d;
// 		// (*acc).x += -0.9 * (*vel).x*(n)*(n);
// 	} 

// 	d = -n*(gridSize.y - (*pos).y) + h;
// 	if(d > 0.0f){
// 		(*pos).y += d*(-0.5f);
// 		(*vel).y -= 1.9*(*vel).y*(-0.5f)*(-0.5f);

// 		// (*acc).y += 10000*(-n)*d;
// 		// (*acc).y += -0.9 * (*vel).y*(-n)*(-n);
// 	} 

// 	d = n*(0.0f - (*pos).y) + h;
// 	if(d > 0.0f){
// 		(*pos).y += d*(0.5f);
// 		(*vel).y -= 1.9*(*vel).y*(0.5f)*(0.5f);

// 		// (*acc).y += 10000*(n)*d;
// 		// (*acc).y += -0.9 * (*vel).y*(n)*(n);
// 	} 

// 	d = -n*(gridSize.z - (*pos).z) + h;
// 	if(d > 0.0f){
// 		(*pos).z += d*(-0.5f);
// 		(*vel).z -= 1.9*(*vel).z*(-0.5f)*(-0.5f);

// 		// (*acc).z += 10000*(-n)*d;
// 		// (*acc).z += -0.9 * (*vel).z*(-n)*(-n);
// 	} 

// 	d = n*(0.0f - (*pos).z) + h;
// 	if(d > 0.0f){
// 		(*pos).z += d*(0.5f);
// 		(*vel).z -= 1.9*(*vel).z*(0.5f)*(0.5f);

// 		// (*acc).z += 10000*(n)*d;
// 		// (*acc).z += -0.9 * (*vel).z*(n)*(n);
// 	} 
// }