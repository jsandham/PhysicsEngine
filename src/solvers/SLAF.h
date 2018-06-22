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

#ifndef SLAF_H
#define SLAF_H

void matrixVectorProduct(const int r[], const int c[], const double v[],
                         const double x[], double y[], const int n);

double dotProduct(const double x[], const double y[], const int n);

double error(const int r[], const int c[], const double v[], const double x[],
             const double b[], const int n);

double fast_error(const int r[], const int c[], const double v[], const double x[],
                  const double b[], const int n, const double tol);

#endif

