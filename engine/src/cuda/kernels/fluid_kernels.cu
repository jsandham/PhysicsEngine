#include "../../../include/cuda/kernels/fluid_kernels.cuh"

#include "../../../include/cuda/helper_math.h"

#include <stdio.h>

using namespace FluidKernels;

#define PI 3.14159265358979323846264f

__device__ const int blockSize = 256;
__device__ const float f1 = 315.0f/(64.0f*PI);
__device__ const float f2 = 45.0f/PI;
__device__ const float nu = 0.05f;// 3.5 * 0.0075f;


__device__ int3 FluidKernels::calcGridPosition(float4 pos, int3 grid, float3 gridSize)
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
__device__ int FluidKernels::calcCellIndex(int3 gridPos, int3 grid)
{
	return grid.y * grid.x * gridPos.z + grid.x * gridPos.y + gridPos.x;
}


// calculate spatial has for infinite domains
__device__ int FluidKernels::calcGridHash(int3 gridPos, int numBuckets)
{
	const uint p1 = 73856093;
	const uint p2 = 19349663;
	const uint p3 = 83492791;
	int n = p1*gridPos.x ^ p2*gridPos.y ^ p3*gridPos.z;
	n %= numBuckets;
	return n;
}

// penalty force boundary conditions http://n-e-r-v-o-u-s.com/education/simulation/week6.php
__device__ void FluidKernels::boundary(float4 *pos, float4 *vel, float h, float3 gridSize)
{
	float d;
	float n = 1.0f;

	d = -n*(gridSize.x - (*pos).x) + h;
	if(d > 0.0f){
		(*pos).x += d*(-0.5f);
		(*vel).x -= 1.9*(*vel).x*(-0.5f)*(-0.5f);

		// (*acc).x += 10000*(-n)*d;
		// (*acc).x += -0.9 * (*vel).x*(-n)*(-n);
	} 

	d = n*(0.0f - (*pos).x) + h;
	if(d > 0.0f){
		(*pos).x += d*(0.5f);
		(*vel).x -= 1.9*(*vel).x*(0.5f)*(0.5f);

		// (*acc).x += 10000*(n)*d;
		// (*acc).x += -0.9 * (*vel).x*(n)*(n);
	} 

	d = -n*(gridSize.y - (*pos).y) + h;
	if(d > 0.0f){
		(*pos).y += d*(-0.5f);
		(*vel).y -= 1.9*(*vel).y*(-0.5f)*(-0.5f);

		// (*acc).y += 10000*(-n)*d;
		// (*acc).y += -0.9 * (*vel).y*(-n)*(-n);
	} 

	d = n*(0.0f - (*pos).y) + h;
	if(d > 0.0f){
		(*pos).y += d*(0.5f);
		(*vel).y -= 1.9*(*vel).y*(0.5f)*(0.5f);

		// (*acc).y += 10000*(n)*d;
		// (*acc).y += -0.9 * (*vel).y*(n)*(n);
	} 

	d = -n*(gridSize.z - (*pos).z) + h;
	if(d > 0.0f){
		(*pos).z += d*(-0.5f);
		(*vel).z -= 1.9*(*vel).z*(-0.5f)*(-0.5f);

		// (*acc).z += 10000*(-n)*d;
		// (*acc).z += -0.9 * (*vel).z*(-n)*(-n);
	} 

	d = n*(0.0f - (*pos).z) + h;
	if(d > 0.0f){
		(*pos).z += d*(0.5f);
		(*vel).z -= 1.9*(*vel).z*(0.5f)*(0.5f);

		// (*acc).z += 10000*(n)*d;
		// (*acc).z += -0.9 * (*vel).z*(n)*(n);
	} 
}


__global__ void FluidKernels::build_spatial_grid
(
	float4 *pos,
	int *particleIndex,
	int *cellIndex,
	int numParticles,
	int3 grid,
	float3 gridSize
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	while (index + offset < numParticles){
		// find which grid cell the particle is in
		int3 gridPos = calcGridPosition(pos[index + offset], grid, gridSize);

		// compute cell index 
		int cindex = calcCellIndex(gridPos, grid);

		particleIndex[index + offset] = index + offset;
		cellIndex[index + offset] = cindex;

		offset += blockDim.x*gridDim.x;
	}
}

__global__ void FluidKernels::reorder_particles
(
	float4 *pos,
	float4 *spos,
	float4 *vel,
	float4 *svel,
	int *particleType,
	int *sparticleType,
	int *cellStartIndex,
	int *cellEndIndex,
	int *cellIndex,
	int *particleIndex,
	int numParticles
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	__shared__ int sharedcellIndex[blockSize + 1];  //blockSize + 1

	while (index + offset < numParticles){
		sharedcellIndex[threadIdx.x] = cellIndex[index + offset];
		if (threadIdx.x == blockDim.x - 1)
		{
			if (index + offset + 1 < numParticles){
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
		int p = particleIndex[index + offset];
		spos[index + offset] = pos[p];
		svel[index + offset] = vel[p];
		sparticleType[index + offset] = particleType[p];

		offset += blockDim.x*gridDim.x;
	}

	__syncthreads();

	if (threadIdx.x == 0 && blockIdx.x == 0){
		cellStartIndex[sharedcellIndex[0]] = 0;
	}
}


__global__ void FluidKernels::calculate_fluid_particle_density
(
	float4 *pos,
	float *rho,
	int *particleType,
	int *cellStartIndex,
	int *cellEndIndex,
	int *cellIndex,
	int *particleIndex,
	int numParticles,
	float h2,
	float h9,
	int3 grid
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;
	int maxIndex = grid.x*grid.y*grid.z;

	// compute density
	while (index + offset < numParticles){

		// fluid particle
		if (particleType[index + offset] == 0){

			// add contributions from particles in cell (and all of its 26 neighour cells)
			float4 position = pos[index + offset];
			float density = 0.0f;

			int c = cellIndex[index + offset];
			for (int k = -1; k <= 1; k++){
				for (int j = -1; j <= 1; j++){
					for (int i = -1; i <= 1; i++){
						int hash = c + grid.y*grid.x*k + grid.x*j + i;
						if (hash >= 0 && hash < maxIndex){

							// loop through all particles in this cell
							for (int l = cellStartIndex[hash]; l < cellEndIndex[hash]; l++){
								float4 p = pos[l];
								float rx = position.x - p.x;
								float ry = position.y - p.y;
								float rz = position.z - p.z;
								float radius2 = rx*rx + ry*ry + rz*rz;
								if (radius2 < h2){
									density += (h2 - radius2) * (h2 - radius2) * (h2 - radius2);
								}
							}
						}
					}
				}
			}

			rho[index + offset] = 0.02f*(f1 / h9) * density;
		}

		offset += blockDim.x * gridDim.x;
	}
}


__global__ void FluidKernels::calculate_solid_particle_density
(
	float4 *pos,
	float *rho,
	int *particleType,
	int *cellStartIndex,
	int *cellEndIndex,
	int *cellIndex,
	int *particleIndex,
	int numParticles,
	float h2,
	float h9,
	int3 grid
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;
	int maxIndex = grid.x*grid.y*grid.z;

	// compute density
	while (index + offset < numParticles){

		// solid particle
		if (particleType[index + offset] == 1){

			float4 position = pos[index + offset];
			int closestIndex = 0;
			float closestRadius = 10.0f;

			// find closest fluid particle
			int c = cellIndex[index + offset];
			for (int k = -1; k <= 1; k++){
				for (int j = -1; j <= 1; j++){
					for (int i = -1; i <= 1; i++){
						int hash = c + grid.y*grid.x*k + grid.x*j + i;
						if (hash >= 0 && hash < maxIndex){

							// loop through all particles in this cell
							for (int l = cellStartIndex[hash]; l < cellEndIndex[hash]; l++){
								float4 p = pos[l];
								float rx = position.x - p.x;
								float ry = position.y - p.y;
								float rz = position.z - p.z;

								float radius = rx*rx + ry*ry + rz*rz;
								if (closestRadius > radius)
								{
									closestRadius = radius;
									closestIndex = index + offset;
								}
							}
						}
					}
				}
			}

			rho[index + offset] = rho[closestIndex];
		}

		offset += blockDim.x * gridDim.x;
	}
}


__global__ void FluidKernels::calculate_pressure
(
	float *rho,
	float *rho0,
	float *pres,
	int numParticles,
	float kappa
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	while (index + offset < numParticles){
		float density = rho[index + offset];

		pres[index + offset] = kappa*(density - rho0[index + offset]) / (density * density);

		offset += blockDim.x * gridDim.x;
	}
}

__global__ void FluidKernels::apply_pressure_and_gravity_acceleration
(
	float4 *pos,
	float4 *vel,
	float *rho,
	float *pres,
	int *particleType,
	int *cellStartIndex,
	int *cellEndIndex,
	int *cellIndex,
	int *particleIndex,
	int numParticles,
	float dt,
	float h,
	float h6,
	int3 grid
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;
	int maxIndex = grid.x*grid.y*grid.z;

	// compute pressure gradient and diffusion force
	while (index + offset < numParticles){

		if (particleType[index + offset] == 0){

			// position, acceleration, and pressure-density squared ratio for this particle
			float4 position = pos[index + offset];
			float prho2 = pres[index + offset];

			float3 acceleration = make_float3(0.0f, 0.0f, 0.0f);

			// add contributions from particles in cell (and all of its 26 neighour cells)
			int c = cellIndex[index + offset];

			for (int k = -1; k <= 1; k++){
				for (int j = -1; j <= 1; j++){
					for (int i = -1; i <= 1; i++){
						int hash = c + grid.y*grid.x*k + grid.x*j + i;
						if (hash >= 0 && hash < maxIndex){

							// loop through all particles in this cell
							/*if (blockIdx.x == 127){
								printf("index + offset: %d  %d  %d  %d %d\n", cellStartIndex[hash], cellEndIndex[hash], cellEndIndex[hash] - cellStartIndex[hash], hash, c);
							}*/


							for (int l = cellStartIndex[hash]; l < cellEndIndex[hash]; l++){
								if (l != index + offset){
									float4 p = pos[l];
									float rx = position.x - p.x;
									float ry = position.y - p.y;
									float rz = position.z - p.z;
									float radius = rx*rx + ry*ry + rz*rz;

									if (radius < h*h){
										radius = sqrt(radius);

										// pressure force contribution
										float presCoefficient = (prho2 + pres[l]) * (h - radius) * (h - radius) / radius;
										acceleration.x += presCoefficient * rx;
										acceleration.y += presCoefficient * ry;
										acceleration.z += presCoefficient * rz;
									}
								}
							}
						}
					}
				}
			}

			acceleration.x = 0.02f * (f2 / h6) * acceleration.x;
			acceleration.y = 0.02f * (f2 / h6) * acceleration.y;
			acceleration.z = 0.02f * (f2 / h6) * acceleration.z;

			vel[index + offset].x += dt*acceleration.x;
			vel[index + offset].y += dt*(-9.81f + acceleration.y);
			vel[index + offset].z += dt*acceleration.z;
		}

		offset += blockDim.x * gridDim.x;
	}
}


__global__ void FluidKernels::compute_solid_particle_velocity
(
	float4 *pos,
	float4 *vel,
	int *particleType,
	int numParticles
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;
	//int maxIndex = grid.x*grid.y*grid.z;

	while (index + offset < numParticles){

		if (particleType[index + offset] == 1){
			vel[index + offset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		}

		offset += blockDim.x * gridDim.x;
	}
}

__global__ void FluidKernels::apply_xsph_viscosity
(
	float4 *pos,
	float4 *vel,
	float *rho,
	int *particleType,
	int *cellStartIndex,
	int *cellEndIndex,
	int *cellIndex,
	int *particleIndex,
	int numParticles,
	float dt,
	float h,
	float h6,
	int3 grid
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;
	int maxIndex = grid.x*grid.y*grid.z;

	// compute pressure gradient and diffusion force
	while (index + offset < numParticles){

		if (particleType[index + offset] == 0){

			// position, acceleration, and pressure-density squared ratio for this particle
			float4 position = pos[index + offset];
			float4 velocity = vel[index + offset];

			float3 xsph_viscosity = make_float3(0.0f, 0.0f, 0.0f);

			// add contributions from particles in cell (and all of its 26 neighour cells)
			int c = cellIndex[index + offset];

			//printf("index+offset: %d  c: %d\n", index+offset, c);

			for (int k = -1; k <= 1; k++){
				for (int j = -1; j <= 1; j++){
					for (int i = -1; i <= 1; i++){
						int hash = c + grid.y*grid.x*k + grid.x*j + i;
						if (hash >= 0 && hash < maxIndex){

							// loop through all particles in this cell
							int start = cellStartIndex[hash];
							int end = cellEndIndex[hash];
							for (int l = start; l < end; l++){
								if (l != index + offset){
									float4 p = pos[l];
									float rx = position.x - p.x;
									float ry = position.y - p.y;
									float rz = position.z - p.z;
									float radius = rx*rx + ry*ry + rz*rz;

									if (radius < h*h){
										radius = sqrt(radius);

										// viscosity force contribution
										float diffCoefficient = (h - radius) / rho[l];

										xsph_viscosity.x += diffCoefficient * (vel[l].x - velocity.x);
										xsph_viscosity.y += diffCoefficient * (vel[l].y - velocity.y);
										xsph_viscosity.z += diffCoefficient * (vel[l].z - velocity.z);
									}
								}
							}
						}
					}
				}
			}

			float density = 0.02f * (f2 / h6) * (nu / rho[index + offset]);

			vel[index + offset].x += density * xsph_viscosity.x; // dont need dt?
			vel[index + offset].y += density * xsph_viscosity.y;
			vel[index + offset].z += density * xsph_viscosity.z;
		}

		offset += blockDim.x * gridDim.x;
	}
}

__global__ void FluidKernels::update_particles
(
	float4 *pos,
	float4 *vel,
	int *particleType,
	float dt,
	float h,
	int numParticles,
	float3 gridSize
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	while (index + offset < numParticles){
		if (particleType[index + offset] == 0){
			float4 my_pos = pos[index + offset];
			float4 my_vel = vel[index + offset];

			my_pos.x += dt*my_vel.x;
			my_pos.y += dt*my_vel.y;
			my_pos.z += dt*my_vel.z;

			boundary(&my_pos, &my_vel, h, gridSize);

			pos[index + offset] = my_pos;
			vel[index + offset] = my_vel;
		}

		offset += blockDim.x * gridDim.x;
	}
}


__global__ void FluidKernels::copy_sph_arrays
(
	float4 *pos,
	float4 *spos,
	float4 *vel,
	float4 *svel,
	int *particleType,
	int *sparticleType,
	float *output,
	int numParticles
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	while (index + offset < numParticles)
	{
		float4 my_pos = pos[index + offset];
		output[3 * (index + offset)] = my_pos.x;
		output[3 * (index + offset) + 1] = my_pos.y;
		output[3 * (index + offset) + 2] = my_pos.z;

		pos[index + offset] = spos[index + offset];
		vel[index + offset] = svel[index + offset];
		particleType[index + offset] = sparticleType[index + offset];

		offset += blockDim.x * gridDim.x;
	}
}






// __global__ void FluidKernels::update_particles2
// (
// 	float4 *pos,
// 	float4 *oldPos,
// 	float4 *vel,
// 	float4 *oldVel,
// 	float dt,
// 	int numParticles,
// 	float3 gridSize
// )
// {
// 	int index = threadIdx.x + blockIdx.x * blockDim.x;
// 	int offset = 0;

// 	while (index + offset < numParticles)
// 	{
// 		float4 my_pos = oldPos[index + offset];
// 		float4 my_vel = oldVel[index + offset];

// 		my_vel.z = my_vel.z - 0.001f * dt;

// 		// new position = old position + velocity * deltaTime
// 		my_pos = my_pos + my_vel * dt;

// 		float bounceDamping = -0.5f;
// 		float particleRadius = 2*0.0078125f;
// 		if (my_pos.x > gridSize.x - particleRadius) { my_pos.x = gridSize.x - particleRadius; my_vel.x *= bounceDamping; }
// 		if (my_pos.x < 0.0f + particleRadius) { my_pos.x = 0.0f + particleRadius; my_vel.x *= bounceDamping; }
// 		if (my_pos.y > gridSize.y - particleRadius) { my_pos.y = gridSize.y - particleRadius; my_vel.y *= bounceDamping; }
// 		if (my_pos.y < 0.0f + particleRadius) { my_pos.y = 0.0f + particleRadius; my_vel.y *= bounceDamping; }
// 		if (my_pos.z > gridSize.z - particleRadius) { my_pos.z = gridSize.z - particleRadius; my_vel.z *= bounceDamping; }
// 		if (my_pos.z < 0.0f + particleRadius) { my_pos.z = 0.0f + particleRadius; my_vel.z *= bounceDamping; }

// 		pos[index + offset] = my_pos;
// 		vel[index + offset] = my_vel;

// 		offset += blockDim.x * gridDim.x;
// 	}
// }


// __global__ void FluidKernels::reorder_particles2
// (
// float4 *pos,
// float4 *oldPos,
// float4 *spos,
// float4 *soldPos,
// float4 *vel,
// float4 *oldVel,
// float4 *svel,
// float4 *soldVel,
// int *particleType,
// int *sparticleType,
// int *cellStartIndex,
// int *cellEndIndex,
// int *cellIndex,
// int *particleIndex,
// int numParticles
// )
// {
// 	int index = threadIdx.x + blockIdx.x * blockDim.x;
// 	int offset = 0;

// 	__shared__ int sharedcellIndex[blockSize + 1];  //blockSize + 1

// 	while (index + offset < numParticles){
// 		sharedcellIndex[threadIdx.x] = cellIndex[index + offset];
// 		if (threadIdx.x == blockDim.x - 1)
// 		{
// 			if (index + offset + 1 < numParticles){
// 				sharedcellIndex[threadIdx.x + 1] = cellIndex[index + offset + 1];
// 			}
// 			else{
// 				sharedcellIndex[threadIdx.x + 1] = -1;
// 			}
// 		}

// 		__syncthreads();

// 		if (sharedcellIndex[threadIdx.x] != sharedcellIndex[threadIdx.x + 1]){
// 			cellStartIndex[sharedcellIndex[threadIdx.x + 1]] = index + offset + 1;
// 			// cellEndIndex[sharedcellIndex[threadIdx.x]] = index + offset;
// 			cellEndIndex[sharedcellIndex[threadIdx.x]] = index + offset + 1;
// 		}

// 		// reorder position and velocity
// 		int p = particleIndex[index + offset];
// 		spos[index + offset] = pos[p];
// 		soldPos[index + offset] = oldPos[p];
// 		svel[index + offset] = vel[p];
// 		soldVel[index + offset] = oldVel[p];
// 		sparticleType[index + offset] = particleType[p];

// 		offset += blockDim.x*gridDim.x;
// 	}

// 	__syncthreads();

// 	if (threadIdx.x == 0 && blockIdx.x == 0){
// 		cellStartIndex[sharedcellIndex[0]] = 0;
// 	}
// }


// // collide two particles using the DEM method from the NVIDIA CUDA SDK sample code
// __device__ float3 FluidKernels::collideParticles(float4 pos1, float4 pos2, float4 vel1, float4 vel2, float radius1, float radius2, float spring, float damping, float shear, float attraction)
// {
// 	// calculate relative position
// 	float3 relPos;
// 	relPos.x = pos2.x - pos1.x;
// 	relPos.y = pos2.y - pos1.y;
// 	relPos.z = pos2.z - pos1.z;

// 	float dist = length(relPos);
// 	float collideDist = radius1 + radius2;

// 	float3 force = make_float3(0.0f);
// 	if (dist < collideDist) {
// 		float3 norm = relPos / dist;

// 		// relative velocity
// 		float3 relVel;
// 		relVel.x = vel2.x - vel1.x;
// 		relVel.y = vel2.y - vel1.y;
// 		relVel.z = vel2.z - vel1.z;

// 		// relative tangential velocity
// 		float3 tanVel = relVel - (dot(relVel, norm) * norm);

// 		// spring force
// 		force = -spring*(collideDist - dist) * norm;
// 		// dashpot (damping) force
// 		force += damping*relVel;
// 		// tangential shear force
// 		force += shear*tanVel;
// 		// attraction
// 		force += attraction*relPos;
// 	}

// 	return force;
// }


// __global__ void	FluidKernels::calculate_collisions
// (
// float4 *pos,
// float4 *oldPos,
// float4 *vel,
// float4 *oldVel,
// int *cellStartIndex,
// int *cellEndIndex,
// int *cellIndex,
// int *particleIndex,
// int numParticles,
// int3 grid
// )
// {
// 	int index = threadIdx.x + blockDim.x * blockIdx.x;
// 	int offset = 0;
// 	int maxIndex = grid.x*grid.y*grid.z;

// 	while (index + offset < numParticles){

// 		float3 force = make_float3(0.0f);
// 		float4 position = oldPos[index + offset];
// 		float4 velocity = oldVel[index + offset];

// 		// add contributions from particles in cell (and all of its 26 neighour cells)
// 		int c = cellIndex[index + offset];

// 		for (int k = -1; k <= 1; k++){
// 			for (int j = -1; j <= 1; j++){
// 				for (int i = -1; i <= 1; i++){
// 					int hash = c + grid.y*grid.x*k + grid.x*j + i;
// 					if (hash >= 0 && hash < maxIndex){

// 						// loop through all particles in this cell
// 						for (int l = cellStartIndex[hash]; l < cellEndIndex[hash]; l++){
// 							if (l != index + offset){
// 								force += collideParticles(position, oldPos[l], velocity, oldVel[l], 2 * 0.0078125f, 2 * 0.0078125f, 0.01f, 0.01f, 0.01f, 0.0f);
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}

// 		//printf("force.x: %f   force.y: %f   force.z: %f  ", force.x, force.y, force.z);
// 		//printf("position %d: %f  %f  %f  ", index + offset, position.x, position.y, position.z);

// 		vel[index + offset] = velocity + make_float4(force, 0.0f);

// 		//printf("force.x: %f  force.y: %f  force.z: %f\n", force.x, force.y, force.z);

// 		offset += blockDim.x * gridDim.x;
// 	}
// }

// __global__ void FluidKernels::copy_arrays
// (
// float4 *pos,
// float4 *oldPos,
// float4 *vel,
// float4 *oldVel,
// int *particleType,
// int *sparticleType,
// float *output,
// int numParticles
// )
// {
// 	int index = threadIdx.x + blockIdx.x * blockDim.x;
// 	int offset = 0;

// 	while (index + offset < numParticles)
// 	{
// 		float4 my_pos = oldPos[index + offset];
// 		output[3 * (index + offset)] = my_pos.x;
// 		output[3 * (index + offset) + 1] = my_pos.y;
// 		output[3 * (index + offset) + 2] = my_pos.z;

// 		/*pos[index + offset] = oldPos[index + offset];
// 		vel[index + offset] = oldVel[index + offset];*/
// 		particleType[index + offset] = sparticleType[index + offset];

// 		offset += blockDim.x * gridDim.x;
// 	}
// }

//__global__ void FluidKernels::copy_arrays
//(
//float4 *pos,
//float4 *oldPos,
//float4 *spos,
//float4 *soldPos,
//float4 *vel,
//float4 *oldVel,
//float4 *svel,
//float4 *soldVel,
//int *particleType,
//int *sparticleType,
//float *output,
//int numParticles
//)
//{
//	int index = threadIdx.x + blockIdx.x * blockDim.x;
//	int offset = 0;
//
//	while (index + offset < numParticles)
//	{
//		float4 my_pos = pos[index + offset];
//		output[3 * (index + offset)] = my_pos.x;
//		output[3 * (index + offset) + 1] = my_pos.y;
//		output[3 * (index + offset) + 2] = my_pos.z;
//
//		pos[index + offset] = spos[index + offset];
//		oldPos[index + offset] = soldPos[index + offset];
//		vel[index + offset] = svel[index + offset];
//		oldVel[index + offset] = soldVel[index + offset];
//		particleType[index + offset] = sparticleType[index + offset];
//
//		offset += blockDim.x * gridDim.x;
//	}
//}



















//__global__ void	FluidKernels::calculate_collisions
//	(
//		float4 *pos,
//		float4 *vel,
//		float4 *acc,
//		int *cellStartIndex,
//		int *cellEndIndex,
//		int *cellIndex,
//		int *particleIndex,
//		int numParticles,
//		int3 grid
//	)
//{
//	int index = threadIdx.x + blockDim.x * blockIdx.x;
//	int offset = 0;
//	int maxIndex = grid.x*grid.y*grid.z;
//
//	while (index + offset < numParticles){
//
//		float3 force = make_float3(0.0f);
//		float4 position = pos[index + offset];
//		float4 velocity = vel[index + offset];
//
//		// add contributions from particles in cell (and all of its 26 neighour cells)
//		int c = cellIndex[index + offset];
//
//		for (int k = -1; k <= 1; k++){
//			for (int j = -1; j <= 1; j++){
//				for (int i = -1; i <= 1; i++){
//					int hash = c + grid.y*grid.x*k + grid.x*j + i;
//					if (hash >= 0 && hash < maxIndex){
//
//						// loop through all particles in this cell
//						for (int l = cellStartIndex[hash]; l < cellEndIndex[hash]; l++){
//							if (l != index + offset){
//								force += collideParticles(position, pos[l], velocity, vel[l], 0.05f, 0.05f, 1.2f, 0.1f, 0.01f, 0.0f);
//							}
//						}
//					}
//				}
//			}
//		}
//
//		//printf("force.x: %f   force.y: %f   force.z: %f  ", force.x, force.y, force.z);
//		//printf("position %d: %f  %f  %f  ", index + offset, position.x, position.y, position.z);
//
//		acc[index + offset] = make_float4(force, 0.0f);
//
//		//printf("force.x: %f  force.y: %f  force.z: %f\n", force.x, force.y, force.z);
//
//		offset += blockDim.x * gridDim.x;
//	}
//}
//
//
//__global__ void FluidKernels::update_particles2
//	(
//		float4 *pos,
//		float4 *vel,
//		float4 *acc,
//		float dt,
//		int numParticles,
//		float3 gridSize
//	)
//{
//	int index = threadIdx.x + blockIdx.x * blockDim.x;
//	int offset = 0;
//
//	while (index + offset < numParticles)
//	{
//		float4 my_pos = pos[index + offset];
//		float4 my_vel = vel[index + offset];
//		float4 my_acc = acc[index + offset];
//
//		/*my_vel.x += my_acc.x;
//		my_vel.y += my_acc.y;
//		my_vel.z += dt*-9.81f + my_acc.z;*/
//
//		my_vel.x += dt*my_acc.x;
//		my_vel.y += dt*my_acc.y;
//		my_vel.z += dt*(-9.81f + my_acc.z);
//
//		my_pos.x += dt*my_vel.x;
//		my_pos.y += dt*my_vel.y;
//		my_pos.z += dt*my_vel.z;
//
//		float bounceDamping = -0.5f;
//		float particleRadius = 0.05f;
//		if (my_pos.x > gridSize.x - particleRadius) { my_pos.x = gridSize.x - particleRadius; my_vel.x *= bounceDamping; }
//		if (my_pos.x < 0.0f + particleRadius) { my_pos.x = 0.0f + particleRadius; my_vel.x *= bounceDamping; }
//		if (my_pos.y > gridSize.y - particleRadius) { my_pos.y = gridSize.y - particleRadius; my_vel.y *= bounceDamping; }
//		if (my_pos.y < 0.0f + particleRadius) { my_pos.y = 0.0f + particleRadius; my_vel.y *= bounceDamping; }
//		if (my_pos.z > gridSize.z - particleRadius) { my_pos.z = gridSize.z - particleRadius; my_vel.z *= bounceDamping; }
//		if (my_pos.z < 0.0f + particleRadius) { my_pos.z = 0.0f + particleRadius; my_vel.z *= bounceDamping; }
//
//		pos[index + offset] = my_pos;
//		vel[index + offset] = my_vel;
//
//		offset += blockDim.x * gridDim.x;
//	}
//}