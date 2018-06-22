//********************************************************************************
//
// IterSolvers: A collection of Iterative Solvers
// Written by James Sandham
// 31 March 2015
//
//********************************************************************************

//********************************************************************************
//
// IterSolvers is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
//********************************************************************************


#include<iostream>
#include"GMRES.h"
#include"SLAF.h"
#include"math.h"


//****************************************************************************
//
// Generalised Minimum Residual
//
//****************************************************************************

#define DEBUG 1


//-------------------------------------------------------------------------------
// generalised minimum residual
//-------------------------------------------------------------------------------
int gmres(const int r[], const int c[], const double v[], double x[], const double b[], 
          const int n, const int restart, const double tol, const int max_iter)
{
  //res = b-A*x and initial error
  double *res = new double[n];
  matrixVectorProduct(r,c,v,x,res,n);
  double err = error(r,c,v,x,b,n);
  for(int i=0;i<n;i++){res[i] = b[i] - res[i];}
  if(err<tol){return 1;}

  //create q and v array
  double *q = new double(n*sizeof(q));
  double *v = new double(n*sizeof(v));
  
  //create H and Q matrices (which are dense and stored as vectors columnwise) 
  double *H = new double(restart*restart*sizeof(H));
  double *Q = new double(n*restart*sizeof(Q));

  //initialize H and Q matrices to zero
  for(int i=0;i<restart*restart;i++){H[i] = 0.0;}
  for(int i=0;i<n*restart;i++){Q[i] = 0.0;}

  double bb = dotProduct(b,b,n);
  for(int i=0;i<n;i++){q[i] = b[i]/sqrt(bb)}
  for(int i=0;i<n;i++){Q[i] = q[i];}

  //gmres 
  int iter = 0, k = 0;
  while(iter<max_iter && err>tol){
    iter++;

    //restart
    while(k<restart){
      k++;

      //Arnoldi iteration
      matrixVectorProduct(r,c,v,q,v,n);
      for(int i=0;i<k;i++){
        H[i+(k-1)*n] = dotProduct(&Q[i*n],v,n); 
        for(int j=0;j<n;j++){
          v[j] = v[j] - H[i+(k-1)*n]*Q[j+i*n]
        }
      }

      double vv = dotProduct(v,v,n);
      H[k+(k-1)*n] = sqrt(vv);
      if(vv<10e-12){
        for(int i=0;i<n;i++){q[i] = 0.0;}
      }
      else{
        for(int i=0;i<n;i++){q[i] = v[i]/H[k+(k-1)*n];}
      }
      for(int i=0;i<n;i++){Q[i+k*n] = q[i];}


      //solve least squares problem h(1:iter+1,1:iter)*y=sqrt(b'*b)*eye(iter+1,1)
      //since H is hessenberg, use givens rotations
      double *R = new double((k+1)*k*sizeof(R));
      for(int i=0;i<k;i++){
        for(int j=0;j<k+1;j++){ 
          R[j+(k+1)*i] = H[j+(restart)*i];
        }
      }
      for(int i=0;i<k+1;i++){x[i] = 0.0;}
      x[0] = sqrt(bb);

      for(int i=0;i<k;i++){
        //Givens 2 by 2 rotation matrix: 
        //  G = [g11,g12
        //       g21,g22]
        double g11=0.0,g12=0.0,g21=0.0,g22=0.0;
        double xi = H[i+i*n], xj = H[i+1+i*n];
        double c,s;
        if(xi<10e-12 && xj<10e-12){
          c = 0.0; s = 0.0;
        }
        else{
          c = xi/sqrt(xi*xi+xj*xj);
          s = -xj/sqrt(xi*xi+xj*xj);
        }
        g11 = c; g22 = c; g12 = -s; g21 = s;
       
      }


      //backward solve
      double *y = new double[k];
      for(int i=k-1;i>-1;i--){
        if(i+1>k-1){
          y[i] = x[i]/R[i+(k+1)*i];
        }
        else{
          for(int j=i+1;j<k;j++){
            y[i] = (x[i]-R[j+(k+1)*i]*y[i])/R[i+(k+1)*i];
          }
        }
      }

      //check error
      
    }
    k = 0;
  }

  delete[] q;
  delete[] v;
  delete[] hr;
  delete[] hc;
  delete[] hv;
  delete[] qr;
  delete[] qc;
  delete[] qv;
}
