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
#include"SLAF.h"
#include"math.h"

//********************************************************************************
//
// Sparse linear algebra functions
//
//********************************************************************************


//-------------------------------------------------------------------------------
// sparse matrix-vector product y = A*x
//-------------------------------------------------------------------------------
void matrixVectorProduct(const int r[], const int c[], const double v[],
                         const double x[], double y[], const int n)
{
  for(int i=0;i<n;i++){
    double s = 0.0;
    for(int j=r[i];j<r[i+1];j++)
      s += v[j]*x[c[j]];
    y[i] = s;
  }
}



//-------------------------------------------------------------------------------
// dot product z = x*y
//-------------------------------------------------------------------------------
double dotProduct(const double x[], const double y[], const int n)
{
  double dot_prod = 0.0;
  for(int i=0;i<n;i++){
    dot_prod = dot_prod + x[i]*y[i];
  }

  return dot_prod;
}




//-------------------------------------------------------------------------------
// error e = |b-A*x|
//-------------------------------------------------------------------------------
double error(const int r[], const int c[], const double v[], const double x[],
             const double b[], const int n)
{
  double e = 0.0;
  for(int j=0;j<n;j++){
    double s = 0.0;
    for(int i=r[j];i<r[j+1];i++)
      s += v[i]*x[c[i]];
    e = e + (b[j] - s)*(b[j] - s);
  }

  return sqrt(e);
}


//-------------------------------------------------------------------------------
// error e = |b-A*x| stops calculating error if error goes above tolerance 
//-------------------------------------------------------------------------------
double fast_error(const int r[], const int c[], const double v[], const double x[],
                  const double b[], const int n, const double tol)
{
  int j = 0;
  double e = 0.0;
  while(e<tol && j<n){
    double s = 0.0;
    for(int i=r[j];i<r[j+1];i++)
      s += v[i]*x[c[i]];
    e = e + (b[j] - s)*(b[j] - s);
    j++;
  }

  return sqrt(e);
}

