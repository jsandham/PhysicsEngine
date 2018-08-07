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


#ifndef AMG_H
#define AMG_H

void amg(const int r[], const int c[], const double v[], double x[], const double b[], 
         const int n, const double theta, const double tol);

void amg_solve(int *ar[], int *ac[], double *av[], double *ad[], int *wr[], int *wc[], 
               double *wv[], double x[], const double b[], int ASizes[], int n1, int n2, 
               int level, int count);

void amg_init(const int r[], const int c[], const double v[], int *ar[], int *ac[], 
              double *av[], double *ad[], int ASizes[], const int n);

int amg_setup(int *ar[], int *ac[], double *av[], double *ad[], int *wr[], int *wc[], 
              double *wv[], int ASizes[], int level, const double theta);

int strength_matrix_size(const int r[], const int c[], const double v[], int rptr_size, 
                         const double theta);

void strength_matrix(const int r[], const int c[], const double v[], int sr[], int sc[], 
                     double sv[], int lambda[], int rptr_size, const double theta);

void strength_transpose_matrix(int sr[], int sc[], double sv[], int str[], int stc[], 
                               double stv[], int lambda[], int rptr_size, const double theta);

void pre_cpoint(int sr[], int sc[], int str[], int stc[], int lambda[], unsigned cfpoints[], 
                int rptr_size);

void pre_cpoint3(int sr[], int sc[], int str[], int stc[], int lambda[], unsigned cfpoints[], 
                 int rptr_size);

void post_cpoint(int sr[], int sc[], unsigned cfpoints[], int rptr_size);

int weight_matrix(const int r[], const int c[], const double v[], double d[], int sr[], 
                  int sc[], double sv[], int *wr[], int *wc[], double *wv[], unsigned cfpoints[], 
                  int rptr_size, int level);

void galerkin_prod(int *ar[], int *ac[], double *av[], int *wr[], int *wc[], double *wv[], 
                   int rptr_size, int m, int level);

void galerkin_prod2(int *ar[], int *ac[], double *av[], double *ad[], int *wr[], int *wc[], 
                    double *wv[], int rptr_size, int m, int level);

void galerkin_prod3(int *ar[], int *ac[], double *av[], int *wr[], int *wc[], double *wv[], 
                    int rptr_size, int m, int level);

void sort(int array1[], double array2[], int start, int end);

int compare_structs(const void *a, const void *b);

#endif
