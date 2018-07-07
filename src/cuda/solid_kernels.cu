#include "solid_kernels.cuh"

const int MAX_NPE = 4;
const int MAX_STIFFNESS = 3;


// __device__ glm::mat4 jacobian(float xi, float et, float ze)
// {
	
// }

__global__ void compute_local_stiffness_matrices
(
	float4* pos,
	float* localStiffnessMatrices,
	int* connect,
	int ne,
	int npe
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	int nodes[MAX_NPE];
	float xpts[MAX_NPE];
	float ypts[MAX_NPE];
	float zpts[MAX_NPE];
	float kmat[MAX_STIFFNESS][MAX_STIFFNESS];

	float jacobian[3][3];

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

		//for(int i = 0; i < numIntPoints; i++){

		//}

		//for(int i=0;i<Npe;i++){
	    //	for(int j=0;j<Npe;j++){
	    //  	emat[i][j] = 0.0;
	    //	}
	  	//}

	  	//double u=1.0, v=1.0, w=1.0;

	  	//QuadratureRule g(nGauss,nDim,Type);
	  	//for(int ip=0;ip<g.nIntPoints;ip++)
	  	//{
	    //	if(nDim>0){u=g.xii[ip];}
	    //	if(nDim>1){v=g.eti[ip];}
	    //	if(nDim>2){w=g.zei[ip];}

	    //	double det = tempDiffMatrix(u,v,w);
	    //	double dv = det*g.wi[ip];

	    //	for(int i=0;i<Npe;i++){
	    //  		for(int j=0;j<Npe;j++){
	    //    		double s = 0.0;
	    //    		for(int k=0;k<nDim;k++)
	    //      			s += bmat[k][i]*bmat[k][j];
	    //    		emat[i][j] += dv*s;
	    //  	}
	    //	}
	  	//}



		offset += blockDim.x * gridDim.x;
	}
}


// double Element::tempDiffMatrix(double xi, double et, double ze)
// {
//   Shape *sh = ShapeFactory::NewShape(Type);

//   (*sh).dshape(xi,et,ze);

//   //set jacobian and inverse jacobian matrix and return jacobian
//   double jac = jacobian(xi,et,ze);

//   //temperature differentiation matrix B
//   if(nDim==1){
//     for(int i=0;i<Npe;i++){
//       bmat[0][i] = imat[0][0]*(*sh).dN[0][i];
//     }
//   }
//   else if(nDim==2){
//     for(int i=0;i<Npe;i++){
//       bmat[0][i] = imat[0][0]*(*sh).dN[0][i]+imat[0][1]*(*sh).dN[1][i];
//       bmat[1][i] = imat[1][0]*(*sh).dN[0][i]+imat[1][1]*(*sh).dN[1][i];
//     }
//   }
//   else if(nDim==3){
//     for(int i=0;i<Npe;i++){
//       bmat[0][i] = imat[0][0]*(*sh).dN[0][i]+imat[0][1]*(*sh).dN[1][i]+imat[0][2]*(*sh).dN[2][i];
//       bmat[1][i] = imat[1][0]*(*sh).dN[0][i]+imat[1][1]*(*sh).dN[1][i]+imat[1][2]*(*sh).dN[2][i];
//       bmat[2][i] = imat[2][0]*(*sh).dN[0][i]+imat[2][1]*(*sh).dN[1][i]+imat[2][2]*(*sh).dN[2][i];
//     }
//   }

//   delete sh;
//   return jac;
// }