#ifndef __CUDA_PHYSICS_CUH__
#define __CUDA_PHYSICS_CUH__

#include <vector>

#include <vector_types.h>

#include "CudaSolvers.cuh"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtx/normal.hpp"

//#include "VoxelGrid.h"

namespace PhysicsEngine
{
	struct CudaCloth
	{
		int nx, ny;

		float dt;
		float kappa;            // spring stiffness coefficient
		float c;                // spring dampening coefficient
		float mass;             // mass

		std::vector<float> particles;
		std::vector<int> particleTypes;

		// used for timing
		float elapsedTime;
		cudaEvent_t start, stop;

		bool initCalled;

		// pointers to host memory
		float4 *h_pos;
		float4 *h_oldPos;
		float4 *h_acc;
		int *h_triangleIndices;
		float *h_triangleVertices;
		float *h_triangleNormals;

		// pointers to device memory
		float4 *d_pos;
		float4 *d_oldPos;
		float4 *d_acc;
		int *d_triangleIndices;
		float *d_triangleVertices;
		float *d_triangleNormals;

		struct cudaGraphicsResource* cudaVertexVBO;
		struct cudaGraphicsResource* cudaNormalVBO;
	};

// have CudaCloth and CudaFEM take a Cloth or Solid as a member?
	struct CudaSolid
	{
		float c;                // specific heat coefficient                         
	    float rho;              // density                            
	    float Q;                // internal heat generation   
	    float k;                // thermal conductivity coefficient

		int dim;                // dimension of mesh (1, 2, or 3) 
	    int ng;                 // number of element groups
	    int n;                  // total number of nodes                      
	    int nte;                // total number of elements (Nte=Ne+Ne_b)       
	    int ne;                 // number of interior elements                
	    int ne_b;               // number of boundary elements                                 
	    int npe;                // number of points per interior element      
	    int npe_b;              // number of points per boundary element      
	    int type;               // interior element type                      
	    int type_b;             // boundary element type    

	    // used for timing
		float elapsedTime;
		cudaEvent_t start, stop;

		bool initCalled;

	    std::vector<float> vertices;
	    std::vector<int> connect;
	    std::vector<int> bconnect;
	    std::vector<int> groups;

	    CudaJacobi jacobi;

		// pointers to host memory
		float4 *h_pos;
		float4 *h_oldPos;
		float4 *h_acc;
		int *h_connect;
		int *h_bconnect;
		int *h_groups;
		int *h_triangleIndices;
		float *h_triangleVertices;
		float *h_triangleNormals;
		float *h_localElementMatrices;

		int *h_rowA;
		int *h_colA;
		float *h_valA;

		// pointers to device memory
		float4 *d_pos;
		float4 *d_oldPos;
		float4 *d_acc;
		int *d_connect;
		int *d_bconnect;
		int *d_groups;
		int *d_triangleIndices;
		float *d_triangleVertices;
		float *d_triangleNormals;
		float *d_localElementMatrices;

		int *d_rowA;
		int *d_colA;
		float *d_valA;

		struct cudaGraphicsResource* cudaVertexVBO;
		struct cudaGraphicsResource* cudaNormalVBO;
	};

	struct CudaFluid
	{
		float h, h2, h6, h9;      // grid parameters
		float dt;                 // TODO: remove. use Physics.time instead
		float kappa;              // kappa
		float rho0;               // particle rest density
		float mass;	              // particle mass
 
		int numParticles;         // number of particles
		int numCells;             // number of cells
		int3 particleGridDim;     // number of voxels in grid for x, y, and z directions 
		float3 particleGridSize;  // size of grid for x, y, and z directions
		//VoxelGrid *grid;

		std::vector<float> particles;
		std::vector<int> particleTypes;

		// used for timing
		float elapsedTime;
		cudaEvent_t start, stop;

		bool initCalled;

		// pointers to host memory
		float4 *h_pos;
		float4 *h_vel;
		float4 *h_acc;
		float4 *h_spos;
		float4 *h_svel;
		float *h_rho, *h_rho0, *h_pres;
		int *h_cellStartIndex;
		int *h_cellEndIndex;
		int *h_cellIndex;
		int *h_particleIndex;
		int *h_particleType;
		int *h_sparticleType;

		// pointers to device memory
		float4 *d_pos;
		float4 *d_vel;
		float4 *d_acc;
		float4 *d_spos;
		float4 *d_svel;
		float *d_rho, *d_rho0, *d_pres;
		float *d_output;
		int *d_cellStartIndex;
		int *d_cellEndIndex;
		int *d_cellHash;
		int *d_particleIndex;
		int *d_particleType;
		int *d_sparticleType;
	};

	class CudaPhysics
	{
		public:
			static void allocate(CudaCloth* cloth);
			static void deallocate(CudaCloth* cloth);
			static void initialize(CudaCloth* cloth);
			static void update(CudaCloth* cloth);

			static void allocate(CudaSolid* fem);
			static void deallocate(CudaSolid* fem);
			static void initialize(CudaSolid* fem);
			static void update(CudaSolid* fem);

			static void allocate(CudaFluid* fluid);
			static void deallocate(CudaFluid* fluid);
			static void initialize(CudaFluid* fluid);
			static void update(CudaFluid* fluid);

		private:
			static void assembleCSR(float* values, int* rowPtrs, int* columns, int* connect, float* localMatrices, int n, int ne, int npe);
	};
}

#endif