#ifndef __QUADRATURERULE_CUH__
#define __QUADRATURERULE_CUH__

class QuadratureRule
{
  public:
    int nDim;
    int nIntPoints;
    float *xii, *eti, *zei;
    float *weights;

  public: 
    __device__ __host__ QuadratureRule(int n, int dim, int typ);
    __device__ __host__ ~QuadratureRule();
};


#endif