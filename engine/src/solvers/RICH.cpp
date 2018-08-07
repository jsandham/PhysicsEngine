//********************************************************************************
//
// IterSolvers: A collection of Iterative Solvers
// Written by James Sandham
// 3 March 2015
//
//********************************************************************************

//********************************************************************************
//
// IterSolvers is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
//********************************************************************************

#include"iostream"
#include"RICH.h"
#include"SLAF.h"
#include"math.h"

//********************************************************************************
//
// Richardson Iteration
//
//********************************************************************************

#define DEBUG 1


//-------------------------------------------------------------------------------
// richardson method
//-------------------------------------------------------------------------------
int rich(const int r[], const int c[], const double v[], double x[], const double b[], 
         const int n, const double theta, const double tol, const int max_iter)
{
  //res = b-A*x and initial error
  double *res = new double[n];
  matrixVectorProduct(r,c,v,x,res,n);
  for(int i=0;i<n;i++){res[i] = b[i] - res[i];}
  double err = error(r,c,v,x,b,n);
  if(err<tol){return 1;}
  

  int iter = 0, inner_iter = 0;
  while(iter<max_iter && err>tol){
    //find res = A*x
    matrixVectorProduct(r,c,v,x,res,n);

    //update approximation
    for(int i=0;i<n;i++){
      x[i] = x[i] + theta*(b[i]-res[i]);
    }

    //calculate error
    if(inner_iter==40){
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

  return iter;
}
