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

#ifndef RICH_H
#define RICH_H

int rich(const int r[], const int c[], const double v[], double x[], const double b[], 
         const int n, const double theta, const double tol, const int max_iter);

#endif
