#ifndef __CUDAPhysicsEngine_CUH__
#define __CUDAPhysicsEngine_CUH__

#include <vector>

#include <vector_types.h>

#include "../glm/glm.hpp"

//#include "VoxelGrid.h"

namespace PhysicsEngine
{
	struct CudaCloth
	{
		int nx, ny;

		std::vector<float> particles;
		std::vector<int> particleTypes;

		// host variables
		float4 *h_pos;
		float4 *h_oldPos;
		float4 *h_acc;
		int *h_triangleIndices;
		float *h_triangleVertices;
		float *h_triangleNormals;

		// device variables
		float4 *d_pos;
		float4 *d_oldPos;
		float4 *d_acc;
		int *d_triangleIndices;
		float *d_triangleVertices;
		float *d_triangleNormals;
		float *d_output;

		// used for timing
		float elapsedTime;
		cudaEvent_t start, stop;

		bool initCalled;

		float dt;
		float kappa;
		float c;
		float mass;

		struct cudaGraphicsResource* vbo_cuda;
	};

	struct CudaFEM
	{
		float c;                //specific heat coefficient                         
	    float rho;              //density                            
	    float Q;                //internal heat generation   
	    float k;                //thermal conductivity coefficient

		int Dim;                //dimension of mesh (1, 2, or 3) 
	    int Ng;                 //number of element groups
	    int N;                  //total number of nodes                      
	    int Nte;                //total number of elements (Nte=Ne+Ne_b)       
	    int Ne;                 //number of interior elements                
	    int Ne_b;               //number of boundary elements                                 
	    int Npe;                //number of points per interior element      
	    int Npe_b;              //number of points per boundary element      
	    int Type;               //interior element type                      
	    int Type_b;             //boundary element type    

	    std::vector<float> vertices;
	    std::vector<int> connect;
	    std::vector<int> bconnect;
	    std::vector<int> groups;

		// host variables
		float4 *h_pos;
		float4 *h_oldPos;
		float4 *h_acc;
		int *h_connect;
		int *h_bconnect;
		int *h_groups;

		// device variables
		float4 *d_pos;
		float4 *d_oldPos;
		float4 *d_acc;
		int *d_connect;
		int *d_bconnect;
		int *d_groups;

		struct cudaGraphicsResource* vbo_cuda;
	};

	struct CudaFluid
	{
		// grid parameters
		float h, h2, h6, h9;

		// particle grid 
		int numParticles;
		int numCells;
		int3 particleGridDim;
		float3 particleGridSize;
		//VoxelGrid *grid;

		std::vector<float> particles;
		std::vector<int> particleTypes;

		// host variables
		float4 *h_pos;
		float4 *h_vel;
		float4 *h_acc;
		float4 *h_spos;
		float4 *h_svel;
		float *h_rho, *h_rho0, *h_pres;
		//float *h_output;
		int *h_cellStartIndex;
		int *h_cellEndIndex;
		int *h_cellIndex;
		int *h_particleIndex;
		int *h_particleType;
		int *h_sparticleType;

		// device variables
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

		// used for timing
		float elapsedTime;
		cudaEvent_t start, stop;

		bool initCalled;
		float dt;
		float kappa;
		float rho0;
		float mass;	
	};

	class CudaPhysics
	{
		public:
			static void allocate(CudaCloth* cloth);
			static void deallocate(CudaCloth* cloth);
			static void initialize(CudaCloth* cloth);
			static void update(CudaCloth* cloth);

			static void allocate(CudaFEM* fem);
			static void deallocate(CudaFEM* fem);
			static void initialize(CudaFEM* fem);
			static void update(CudaFEM* fem);

			static void allocate(CudaFluid* fluid);
			static void deallocate(CudaFluid* fluid);
			static void initialize(CudaFluid* fluid);
			static void update(CudaFluid* fluid);
	};
}

#endif