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

#ifndef PCG_H
#define PCG_H

int pcg(const int r[], const int c[], const double v[], double x[], const double b[], 
        const int n, const double tol, const int max_iter);

int pcg2(const int r[], const int c[], const double v[], double x[], const double b[], 
         const int n, const double tol, const int max_iter);

#endif
