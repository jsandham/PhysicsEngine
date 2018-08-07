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

#ifndef JGS_H
#define JGS_H

int jac(const int r[], const int c[], const double v[], double x[], const double b[], 
        const int n, const double tol, const int max_iter);

int gs(const int r[], const int c[], const double v[], double x[], const double b[], 
       const int n, const double tol, const int max_iter);

int sor(const int r[], const int c[], const double v[], double x[], const double b[], 
        const int n, const double omega, const double tol, const int max_iter);

int sgs(const int r[], const int c[], const double v[], double x[], const double b[], 
        const int n, const double tol, const int max_iter);

int ssor(const int r[], const int c[], const double v[], double x[], const double b[], 
         const int n, const double omega, const double tol, const int max_iter);

#endif
