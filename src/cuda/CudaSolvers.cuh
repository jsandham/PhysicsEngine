#ifndef __CUDA_SOLVERS_H__
#define __CUDA_SOLVERS_H__

namespace PhysicsEngine
{
	// could instead have them take in something like:
	// struct CRSJacobi
	// {
    //		fill in matrix, solution data etc
	// } 

	// this method might be better for pcg and amg where we need to allocate other data on the gpu in addition to the obvious stuff shown below
	// it would also make it more consistent with CudaPhysics...

	struct CudaJacobi
	{
		// device variables
		float* d_xnew;
		float* d_diag;
	};

	struct CudaPCG
	{
		// device variables
		float* d_r;
		float* d_z;
		float* d_h;
		float* sold;
		float* snew;
		float* d_diag;
	};

	struct CudaAMG
	{
		
	};


	class CudaSolvers
	{
		public:
			static int jacobi(int* d_row, int* d_col, float* d_val, float* d_x, float* d_b, int n, int maxIter, float tol, CudaJacobi* jacobi);
			static int pcg(int* d_row, int* d_col, float* d_val, float* d_x, float* d_b, int n, int maxIter, float tol, CudaPCG* pcg);
			static int amg(int* d_row, int* d_col, float* d_val, float* d_x, float* d_b, int n, float theta, float tol, CudaAMG* amg);
	};
}

#endif