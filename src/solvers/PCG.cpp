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

#include <iostream>
#include "AMG.h"
#include "PCG.h"
#include "SLAF.h"
#include "math.h"


//****************************************************************************
//
// Preconditioned Conjugate Gradient 
//
//****************************************************************************

#define DEBUG 1


//-------------------------------------------------------------------------------
// preconditioned conjugate gradient
//-------------------------------------------------------------------------------
int pcg(const int r[], const int c[], const double v[], double x[], const double b[], 
        const int n, const double tol, const int max_iter)
{
  //res = b-A*x and initial error
  double *res = new double[n];
  matrixVectorProduct(r,c,v,x,res,n);
  double err = error(r,c,v,x,b,n);
  for(int i=0;i<n;i++){res[i] = b[i] - res[i];}
  if(err<tol){return 1;}

  //create z and p vector
  double *z = new double[n];
  double *p = new double[n];

  //z = (M^-1)*r
  for(int i=0;i<n;i++)
    z[i] = res[i];

  //p = z
  for(int i=0;i<n;i++)
    p[i] = z[i];

  int iter = 0, inner_iter = 0;
  while(iter<max_iter && err>tol){
    //z = A*p and alpha = (z,r)/(Ap,p)
    double alpha = 0.0, alpha1 = 0.0, alpha2 = 0.0;
    for(int i=0;i<n;i++){alpha1 += z[i]*res[i];}
    matrixVectorProduct(r,c,v,p,z,n);
    for(int i=0;i<n;i++){alpha2 += z[i]*p[i];}
    alpha = alpha1/alpha2;

    //update x and res
    for(int i=0;i<n;i++){
      x[i] += alpha*p[i];
      res[i] -= alpha*z[i];
    }

    //z = (M^-1)*r
    for(int i=0;i<n;i++)
      z[i] = res[i];

    //find beta
    double beta = 0.0;
    for(int i=0;i<n;i++){beta += z[i]*res[i];}
    beta = -beta/alpha1;

    //update p
    for(int i=0;i<n;i++)
      p[i] = z[i] - beta*p[i];

    //calculate error
    if(inner_iter==100){
      err = error(r,c,v,x,b,n);
      inner_iter = 0;
      #if(DEBUG)
        std::cout<<"error: "<<err<<std::endl;
      #endif
    }
    iter++;
    inner_iter++;
  }

  delete[] res;
  delete[] z;
  delete[] p;
  return iter;
}














































//-------------------------------------------------------------------------------
// preconditioned conjugate gradient
//-------------------------------------------------------------------------------
int pcg2(const int r[], const int c[], const double v[], double x[], const double b[], 
         const int n, const double tol, const int max_iter)
{
  //res = b-A*x and initial error
  double *res = new double[n];
  matrixVectorProduct(r,c,v,x,res,n);
  double err = error(r,c,v,x,b,n);
  for(int i=0;i<n;i++){res[i] = b[i] - res[i];}
  if(err<tol){return 1;}

  //create w and p vector
  double *w = new double[n];
  double *p = new double[n];

  //AMG preconditioner setup
  //int level = 10;
  //int *matrixSizes = new int[level+1];   // size of A matrix at each level
  //int **ar = new int*[level+1];          //
  //int **ac = new int*[level+1];          // pointers to A-matrix at each level
  //double **av = new double*[level+1];    // and diagonal entries of A-matrix
  //double **ad = new double*[level+1];    //
  //int **wr = new int*[level];            //
  //int **wc = new int*[level];            // pointers to W-matrix at each level
  //double **wv = new double*[level];      //

  //amg_init(r,c,v,ar,ac,av,ad,matrixSizes,n);
  //level = amg_setup(ar,ac,av,ad,wr,wc,wv,matrixSizes,level,0.25);

  int iter = 0, inner_iter = 0;
  double gamma0 = 1.0, gammai = 1.0;
  double omega=1.1, omega2=2.0-omega;
  while(iter<max_iter && err>tol){
    //w = (M^-1)*r
    for(int i=0;i<n;i++){
    //  //w[i] = md[i]*r[i];
      w[i] = res[i];
    }   
    //amg_solve(ar,ac,av,ad,wr,wc,wv,w,res,matrixSizes,2,2,level,0);

    //gam = (r,w)
    double gammai1 = gammai;
    gammai = 0;
    for(int i=0;i<n;i++){gammai += res[i]*w[i];}
    if(iter==0){
      gamma0 = gammai;
      for(int i=0;i<n;i++)
        p[i] = w[i];
    }
    else{
      double rg = gammai/gammai1;
      for(int i=0;i<n;i++)
        p[i] = w[i] + rg*p[i];
    }

    //w = A*p
    matrixVectorProduct(r,c,v,p,w,n);
    double beta = 0;
    for(int i=0;i<n;i++)
      beta += p[i]*w[i];
    double alpha = gammai/beta;

    //update x and res
    for(int i=0;i<n;i++){
      x[i] += alpha*p[i];
      res[i] -= alpha*w[i];
    }
    
    //calculate error
    if(inner_iter==1){
      err = error(r,c,v,x,b,n);
      inner_iter = 0;
      #if(DEBUG)
        std::cout<<"error: "<<err<<std::endl;
      #endif
    }
    iter++;
    inner_iter++;
  }

  delete[] res;
  delete[] w;
  delete[] p;
  return iter;
}





//-------------------------------------------------------------------------------
// diagonal preconditioner md = diag(A)^-1
//-------------------------------------------------------------------------------
//void diagonalPreconditioner()
//{
//  for(int j=0;j<neq;j++){
//    int i;
//    for(i=nrow[j];i<nrow[j+1];i++)
//      if(ncol[i]==j) break;
//    md[j] = 1.0/A[i];
//    if(A[i]==0){std::cout<<"i: "<<i<<std::endl;}
//  }
//}





