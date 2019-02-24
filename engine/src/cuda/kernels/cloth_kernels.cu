#include "../../../include/cuda/kernels/cloth_kernels.cuh"

#include "../../../include/cuda/helper_math.h"

#include <stdio.h>

using namespace ClothKernels;

__device__ int2 nextNeigh(int n)
{
	if (n == 0)		return make_int2(-1, -1);
	if (n == 1)		return make_int2( 0, -1);
	if (n == 2)		return make_int2( 1, -1);
	if (n == 3)		return make_int2( 1,  0);
	if (n == 4)		return make_int2( 1,  1);
	if (n == 5)		return make_int2( 0,  1);
	if (n == 6)		return make_int2(-1,  1);
	if (n == 7)		return make_int2(-1,  0);

	if (n == 8)		return make_int2(-2, -2);
	if (n == 9)		return make_int2( 2, -2);
	if (n == 10)	return make_int2( 2,  2);
	if (n == 11)	return make_int2(-2,  2);
	
	return make_int2(0, 0);
}

__global__ void ClothKernels::calculate_forces
(
	float4 *pos,
	float4 *oldPos,
	float4 *acc,
	float mass,
	float kappa,
	float c,
	float dt,
	int nx,
	int ny
)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	int index = ix + nx*iy;

	int x[12] = { 0, 0, -1, 1, -1, 1, -1, 1, 0, 0, 2, -2 };
	int y[12] = { -1, 1, 0, 0, -1, -1, 1, 1, 2, -2, 0, 0 };
	float rest[12] = { 1.0f, 1.0f, 1.0f, 1.0f, sqrt(2.0f), sqrt(2.0f), sqrt(2.0f), sqrt(2.0f), 2.0f, 2.0f, 2.0f, 2.0f };

	float3 force = make_float3(0.0f, -9.81f, 0.0f);

	float3 position = make_float3(pos[index].x, pos[index].y, pos[index].z);
	float3 oldPosition = make_float3(oldPos[index].x, oldPos[index].y, oldPos[index].z);
	float3 velocity = (position - oldPosition) / dt;

	// spring forces
	for (int k = 0; k < 12; k++){
		int i = x[k];
		int j = y[k];

		if ((iy + j) < 0 || (iy + j) > (ny - 1)){
			continue;
		}

		if ((ix + i) < 0 || (ix + i) > (nx - 1)){
			continue;
		}

		float4 temp1 = pos[index + nx*j + i];
		//float4 temp2 = oldPos[index + nx*j + i];
		float3 p = make_float3(temp1.x, temp1.y, temp1.z);
		//float3 v = make_float3((temp1.x - temp2.x) / dt, (temp1.y - temp2.y) / dt, (temp1.z - temp2.z) / dt);
		float3 diff = p - position;
		//float3 vdiff = v - velocity;

		float restLength = rest[k] * 0.005f;

		//force += kappa * (normalize(diff) * (length(diff) - restLength)) - c*dot(vdiff, diff) / length(diff);// c * velocity;
		force += kappa * (normalize(diff) * (length(diff) - restLength)) - c * velocity;
	}

	if (iy == 0){// && (ix == 0 || ix == 255)){
		force = make_float3(0.0f, 0.0f, 0.0f);
	}

	acc[index].x = mass * force.x;
	acc[index].y = mass * force.y;
	acc[index].z = mass * force.z;
}

__global__ void ClothKernels::verlet_integration
(
	float4 *pos,
	float4 *oldPos,
	float4 *acc,
	float dt,
	int nx,
	int ny
)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	int index = ix + nx*iy;

	float4 my_pos = pos[index];
	float4 my_oldPos = oldPos[index];
	float4 my_acc = acc[index];

	float4 tmp = my_pos;
	my_pos = 2.0f * my_pos - my_oldPos + my_acc * dt * dt;
	my_oldPos = tmp;

	// float3 p = make_float3(my_pos.x, my_pos.y, my_pos.z);
	// float3 center = make_float3(0.6f, 0.5f, 0.5f);
	// float radius = 0.25f;

	// if (length(p - center) < radius)
	// {
	// 	// collision
	// 	float3 coll_dir = normalize(p - center);
	// 	p = center + coll_dir * radius;
	// }

	// my_pos.x = p.x;
	// my_pos.y = p.y;
	// my_pos.z = p.z;

	// if (my_pos.y  < 0.0)
	// {
	// 	my_pos.y = 0.0;
	// 	//my_oldPos += (my_pos - my_oldPos) * 0.03;
	// }

	pos[index] = my_pos;
	oldPos[index] = my_oldPos;

	//output[3 * index] = my_pos.x;
	//output[3 * index + 1] = my_pos.y;
	//output[3 * index + 2] = my_pos.z;
}

__global__ void ClothKernels::update_triangle_mesh
(
	float4 *pos,
	int *triangleIndices,
	float *triangleVertices,
	int nx,
	int ny
)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	int index = ix + nx*iy;

	while(index < 2*(nx-1)*(ny-1)){
		int ind1 = triangleIndices[3*index];
		int ind2 = triangleIndices[3*index + 1];
		int ind3 = triangleIndices[3*index + 2];

		triangleVertices[9*index] = pos[ind1].x;
		triangleVertices[9*index + 1] = pos[ind1].y;
		triangleVertices[9*index + 2] = pos[ind1].z;

		triangleVertices[9*index + 3] = pos[ind2].x;
		triangleVertices[9*index + 4] = pos[ind2].y;
		triangleVertices[9*index + 5] = pos[ind2].z;

		triangleVertices[9*index + 6] = pos[ind3].x;
		triangleVertices[9*index + 7] = pos[ind3].y;
		triangleVertices[9*index + 8] = pos[ind3].z;

		index += 256*256;
	}
}








__global__ void ClothKernels::apply_constraints
(
	float4 *pos,
	int nx,
	int ny
)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (iy == 0 && (ix == 0 || ix == 255)){
		return;
	}

	int index = ix + nx*iy;

	int x[12] = { 0, 0, -1, 1, -1, 1, -1, 1, 0, 0, 2, -2 };
	int y[12] = { -1, 1, 0, 0, -1, -1, 1, 1, 2, -2, 0, 0 };
	float rest[12] = { 1.0f, 1.0f, 1.0f, 1.0f, sqrt(2.0f), sqrt(2.0f), sqrt(2.0f), sqrt(2.0f), 2.0f, 2.0f, 2.0f, 2.0f };

	float3 position = make_float3(pos[index].x, pos[index].y, pos[index].z);

	// spring forces
	for (int k = 0; k < 12; k++){
		int i = x[k];
		int j = y[k];

		if ((iy + j) < 0 || (iy + j) > (ny - 1)){
			continue;
		}

		if ((ix + i) < 0 || (ix + i) > (nx - 1)){
			continue;
		}

		float4 temp1 = pos[index + nx*j + i];
		float3 p = make_float3(temp1.x, temp1.y, temp1.z);
		float3 diff = p - position;

		float restLength = rest[k] * 0.005f;

		//float delta = fmaxf(restLength - length(diff), fminf(restLength, length(diff) - restLength));
		float delta = fminf(fmaxf(length(diff), 0.0f), 1.2f*restLength);

		position = p - delta * normalize(diff);
	}

	float4 adjustedPos = make_float4(position.x, position.y, position.z, 0.0f);

	pos[index] = adjustedPos;
}












// __global__ void ClothKernels::verlet(	float4 * g_pos_in, float4 * g_pos_old_in, float4 * g_pos_out, float4 * g_pos_old_out, 
// 							int side, float stiffness, float damp, float inverse_mass, int coll_primitives )
// {
//     uint index = blockIdx.x * blockDim.x + threadIdx.x;
// 	int ix = index % side; 
// 	int iy = index / side; 

// 	volatile float4 posData = g_pos_in[index];    // ensure coalesced read
//     volatile float4 posOldData = g_pos_old_in[index];

//     float3 pos = make_float3(posData.x, posData.y, posData.z);
//     float3 pos_old = make_float3(posOldData.x, posOldData.y, posOldData.z);
// 	float3 vel = (pos - pos_old) / 0.01;

// 	float3 force = make_float3(0.0, -9.81, 0.0);
// 	float inv_mass = inverse_mass;
// 	if (index <= (side - 1.0))
// 		inv_mass = 0.f;

// 	float step = 1.0 / (side - 1.0);

// 	for (int k = 0; k < 12; k++)
// 	{
// 		int2 coord = nextNeigh(k);
// 		int j = coord.x;
// 		int i = coord.y;

// 		if (((iy + i) < 0) || ((iy + i) > (side - 1)))
// 			continue;

// 		if (((ix + j) < 0) || ((ix + j) > (side - 1)))
// 			continue;

// 		int index_neigh = (iy + i) * side + ix + j;

// 		volatile float4 pos_neighData = g_pos_in[index_neigh];

// 		float3 pos_neigh = make_float3(pos_neighData.x, pos_neighData.y, pos_neighData.z);

// 		float3 diff = pos_neigh - pos;

// 		float3 curr_diff = diff;	// curr diff is the normalized direction of the spring
// 		curr_diff = normalize(curr_diff);
		
// 		last_diff = curr_diff;

// 		float2 fcoord = make_float2(coord)* step;
// 		float rest_length = length(fcoord);

// 		force += (curr_diff * (length(diff) - rest_length)) * stiffness - vel * damp;
// 	}

// 	float3 acc = make_float3(0, 0, 0);
// 	acc = force * inv_mass;

// 	// verlet
// 	float3 tmp = pos; 
// 	pos = pos * 2 - pos_old + acc * 0.01 * 0.01;
// 	pos_old = tmp;

// 	syncthreads();

// 	g_pos_out[index] = make_float4(pos.x, pos.y, pos.z, posData.w);
// 	g_pos_old_out[index] = make_float4(pos_old.x, pos_old.y, pos_old.z, posOldData.w);

// }