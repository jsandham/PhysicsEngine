#ifndef __SOLIDDEVICEDATA_H__
#define __SOLIDDEVICEDATA_H__ 

#include <vector>

#include <vector_types.h>

#include <cuda.h>
//#include <cudagl.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "kernels/solid_kernels.cuh"

#include "CudaSolvers.cuh"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtx/normal.hpp"

namespace PhysicsEngine
{
	// have CudaCloth and CudaFEM take a Cloth or Solid as a member?
	struct SolidDeviceData
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

	void allocateSolidDeviceData(SolidDeviceData* solid);
	void deallocateSolidDeviceData(SolidDeviceData* solid);
	void initializeSolidDeviceData(SolidDeviceData* solid);
	void updateSolidDeviceData(SolidDeviceData* solid);

	void assembleCSR(float* values, int* rowPtrs, int* columns, int* connect, float* localMatrices, int n, int ne, int npe);

}

#endif