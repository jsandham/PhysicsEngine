#include <stdio.h>

#include "solid_kernels.cuh"

const int MAX_NPE = 4;
const int MAX_DIM = 3;

__constant__ float tet3D4_xii[4] = {0.5854101966249685f, 0.1381966011250105f, 0.1381966011250105f, 0.1381966011250105f};
__constant__ float tet3D4_eti[4] = {0.1381966011250105f, 0.1381966011250105f, 0.1381966011250105f, 0.5854101966249685f}; 
__constant__ float tet3D4_zei[4] = {0.1381966011250105f, 0.1381966011250105f, 0.5854101966249685f, 0.1381966011250105f};
__constant__ float tet3d4_w[4] = {0.041666666666666f, 0.041666666666666f, 0.041666666666666f, 0.041666666666666f};

__constant__ float tet3D5_xii[5] = {0.2500000000000000f, 0.5000000000000000f, 0.1666666666666667f, 0.1666666666666667f, 0.1666666666666667f};
__constant__ float tet3D5_eti[5] = {0.2500000000000000f, 0.1666666666666667f, 0.1666666666666667f, 0.1666666666666667f, 0.5000000000000000f}; 
__constant__ float tet3D5_zei[5] = {0.2500000000000000f, 0.1666666666666667f, 0.1666666666666667f, 0.5000000000000000f, 0.1666666666666667f};
__constant__ float tet3d5_w[5] = {-0.13333333333333f, 0.075000000000000f, 0.075000000000000f, 0.075000000000000f, 0.075000000000000f};



__global__ void compute_local_stiffness_matrices
(
	float4* pos,
	float* localStiffnessMatrices,
	int* connect,
	int ne,
	int npe,
	int type
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	int nodes[MAX_NPE];
	float xpts[MAX_NPE];
	float ypts[MAX_NPE];
	float zpts[MAX_NPE];
	float kmat[MAX_NPE][MAX_NPE];
	float bmat[MAX_DIM][MAX_NPE];

	//float N[MAX_NPE];
	float dN[MAX_DIM][MAX_NPE];

	float jmat[MAX_DIM][MAX_DIM];
	float inv_jmat[MAX_DIM][MAX_DIM];

	int nIntPoints;
	float *xii;
	float *eti;
	float *zei;
	float *weights;

 	// all threads should take the same path
	switch(type)
	{
		case 4:
			nIntPoints = 4;
			xii = tet3D4_xii;
			eti = tet3D4_eti;
			zei = tet3D4_zei;
			weights = tet3d4_w;
			break;
		case 5:
			nIntPoints = 5;
			xii = tet3D5_xii;
			eti = tet3D5_eti;
			zei = tet3D5_zei;
			weights = tet3d5_w;
	}

	while(index + offset < ne){

		for(int i = 0; i < npe; i++){
			for(int j = 0; j < npe; j++){
				kmat[i][j] = 0.0f;
			}
		}

		for(int i = 0; i < npe; i++){
			nodes[i] = connect[npe*(index + offset) + i];
			xpts[i] = pos[nodes[i] - 1].x;
			ypts[i] = pos[nodes[i] - 1].y;
			zpts[i] = pos[nodes[i] - 1].z;
		}

		// loop over all integration points
		for(int ip = 0; ip < nIntPoints; ip++){
			
			// float xi = xii[ip];
			// float et = eti[ip];
			// float ze = zei[ip];

			// compute shape functions and derivatives of shape functions at (xi, et, ze)
			// N[0] = xi;
			// N[1] = et;
			// N[2] = ze;
			// N[3] = 1-xi-et-ze;

		  	//derivatives of shape functions for linear 3D tetrahedra at node (xi,et,ze)
			dN[0][0] = 1; dN[0][1] = 0; dN[0][2] = 0; dN[0][3] = -1;
			dN[1][0] = 0; dN[1][1] = 1; dN[1][2] = 0; dN[1][3] = -1;
			dN[2][0] = 0; dN[2][1] = 0; dN[2][2] = 1; dN[2][3] = -1;

			// compute jacobian and inverse jacobian matrix
			for(int i = 0; i < npe; i++){
				jmat[0][0] += dN[0][i]*xpts[i];
			    jmat[0][1] += dN[0][i]*ypts[i];
			    jmat[0][2] += dN[0][i]*zpts[i];
			    jmat[1][0] += dN[1][i]*xpts[i];
			    jmat[1][1] += dN[1][i]*ypts[i];
			    jmat[1][2] += dN[1][i]*zpts[i];
			    jmat[2][0] += dN[2][i]*xpts[i];
			    jmat[2][1] += dN[2][i]*ypts[i];
			    jmat[2][2] += dN[2][i]*zpts[i];
			}

		    float jac = jmat[0][0]*(jmat[1][1]*jmat[2][2] - jmat[2][1]*jmat[1][2])
			        - jmat[0][1]*(jmat[1][0]*jmat[2][2] - jmat[2][0]*jmat[1][2])
			        + jmat[0][2]*(jmat[1][0]*jmat[2][1] - jmat[1][1]*jmat[2][0]);
		    inv_jmat[0][0] = (jmat[1][1]*jmat[2][2] - jmat[2][1]*jmat[1][2]) / jac;
		    inv_jmat[0][1] = (jmat[0][2]*jmat[2][1] - jmat[0][1]*jmat[2][2]) / jac;
		    inv_jmat[0][2] = (jmat[0][1]*jmat[1][2] - jmat[0][2]*jmat[1][1]) / jac;
		    inv_jmat[1][0] = (jmat[1][2]*jmat[2][0] - jmat[1][0]*jmat[2][2]) / jac;
		    inv_jmat[1][1] = (jmat[0][0]*jmat[2][2] - jmat[0][2]*jmat[2][0]) / jac;
		    inv_jmat[1][2] = (jmat[0][2]*jmat[1][0] - jmat[0][0]*jmat[1][2]) / jac;
		    inv_jmat[2][0] = (jmat[1][0]*jmat[2][1] - jmat[1][1]*jmat[2][0]) / jac;
		    inv_jmat[2][1] = (jmat[0][1]*jmat[2][0] - jmat[0][0]*jmat[2][1]) / jac;
		    inv_jmat[2][2] = (jmat[0][0]*jmat[1][1] - jmat[0][1]*jmat[1][0]) / jac;

			// compute displacement differentiation matrix bmat
		    for(int i = 0; i < npe; i++){
		      bmat[0][i] = inv_jmat[0][0]*dN[0][i] + inv_jmat[0][1]*dN[1][i] + inv_jmat[0][2]*dN[2][i];
		      bmat[1][i] = inv_jmat[1][0]*dN[0][i] + inv_jmat[1][1]*dN[1][i] + inv_jmat[1][2]*dN[2][i];
		      bmat[2][i] = inv_jmat[2][0]*dN[0][i] + inv_jmat[2][1]*dN[1][i] + inv_jmat[2][2]*dN[2][i];
		    }

			float dv = jac * weights[ip];

			for(int i = 0; i < npe; i++){
				for(int j = 0; j < npe; j++){
					float s = 0.0f;
					for(int k = 0; k < 3; k++){
						s += bmat[k][i]*bmat[k][j];
					}

					kmat[i][j] += dv * s;
				}
			}
		}

		// write local stiffness matrix (kmat) to global memory array
		for(int i = 0; i < npe; i++){
			for(int j = 0; j < npe; j++){
				int k = npe*npe*(index + offset) + npe*i + j;
				localStiffnessMatrices[k] = kmat[i][j];
			}
		}

		offset += blockDim.x * gridDim.x;
	}
}

__global__ void compute_local_mass_matrices
(
	float4* pos,
	float* localMassMatrices,
	int* connect,
	int ne,
	int npe,
	int type
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	int nodes[MAX_NPE];
	float xpts[MAX_NPE];
	float ypts[MAX_NPE];
	float zpts[MAX_NPE];
	float mmat[MAX_NPE][MAX_NPE];

	float N[MAX_NPE];
	float dN[MAX_DIM][MAX_NPE];

	float jmat[MAX_DIM][MAX_DIM];

	int nIntPoints;
	float *xii;
	float *eti;
	float *zei;
	float *weights;

 	// all threads should take the same path
	switch(type)
	{
		case 4:
			nIntPoints = 4;
			xii = tet3D4_xii;
			eti = tet3D4_eti;
			zei = tet3D4_zei;
			weights = tet3d4_w;
			break;
		case 5:
			nIntPoints = 5;
			xii = tet3D5_xii;
			eti = tet3D5_eti;
			zei = tet3D5_zei;
			weights = tet3d5_w;
	}

	while(index + offset < ne){

		for(int i = 0; i < npe; i++){
			for(int j = 0; j < npe; j++){
				mmat[i][j] = 0.0f;
			}
		}

		for(int i = 0; i < npe; i++){
			nodes[i] = connect[npe*(index + offset) + i];
			xpts[i] = pos[nodes[i] - 1].x;
			ypts[i] = pos[nodes[i] - 1].y;
			zpts[i] = pos[nodes[i] - 1].z;
		}

		// loop over all integration points
		for(int ip = 0; ip < nIntPoints; ip++){
			
			float xi = xii[ip];
			float et = eti[ip];
			float ze = zei[ip];

			// compute shape functions and derivatives of shape functions at (xi, et, ze)
			N[0] = xi;
		  	N[1] = et;
		 	N[2] = ze;
		  	N[3] = 1-xi - et - ze;

		  	//derivatives of shape functions for linear 3D tetrahedra at node (xi,et,ze)
			dN[0][0] = 1; dN[0][1] = 0; dN[0][2] = 0; dN[0][3] = -1;
			dN[1][0] = 0; dN[1][1] = 1; dN[1][2] = 0; dN[1][3] = -1;
			dN[2][0] = 0; dN[2][1] = 0; dN[2][2] = 1; dN[2][3] = -1;

			// compute jacobian matrix
			for(int i = 0; i < npe; i++){
				jmat[0][0] += dN[0][i]*xpts[i];
			    jmat[0][1] += dN[0][i]*ypts[i];
			    jmat[0][2] += dN[0][i]*zpts[i];
			    jmat[1][0] += dN[1][i]*xpts[i];
			    jmat[1][1] += dN[1][i]*ypts[i];
			    jmat[1][2] += dN[1][i]*zpts[i];
			    jmat[2][0] += dN[2][i]*xpts[i];
			    jmat[2][1] += dN[2][i]*ypts[i];
			    jmat[2][2] += dN[2][i]*zpts[i];
			}

		    float jac = jmat[0][0]*(jmat[1][1]*jmat[2][2] - jmat[2][1]*jmat[1][2])
			        - jmat[0][1]*(jmat[1][0]*jmat[2][2] - jmat[2][0]*jmat[1][2])
			        + jmat[0][2]*(jmat[1][0]*jmat[2][1] - jmat[1][1]*jmat[2][0]);

			float dv = jac * weights[ip];

			for(int i = 0; i < npe; i++){
				for(int j = 0; j < npe; j++){
					mmat[i][j] += dv*N[i]*N[j];
				}
			}
		}

		// write local mass matrix (mmat) to global memory array
		for(int i = 0; i < npe; i++){
			for(int j = 0; j < npe; j++){
				int k = npe*npe*(index + offset) + npe*i + j;
				localMassMatrices[k] = mmat[i][j];
			}
		}

		offset += blockDim.x * gridDim.x;
	}
}