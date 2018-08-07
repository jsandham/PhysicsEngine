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
#include <stdlib.h>
#include "AMG.h"
#include "SLAF.h"
#include "math.h"
#include "debug.h"

//********************************************************************************
//
// AMG: Classical Algebraic Multigrid
//
//********************************************************************************

#define FAST_ERROR 0
#define MAX_VCYCLES 50


//-------------------------------------------------------------------------------
// structure representing an array
//-------------------------------------------------------------------------------
struct array{
  int value;
  unsigned id;
};


//-------------------------------------------------------------------------------
// AMG main function
//-------------------------------------------------------------------------------
void amg(const int r[], const int c[], const double v[], double x[], const double b[], 
         const int n, const double theta, const double tol)
{
  int n1 = 2;                            //
  int n2 = 1;                            // number of smoothing steps and maximum number of levels
  int level = 10;                        //

  int *matrixSizes = new int[level+1];   // size of A matrix at each level 

  int **ar = new int*[level+1];          //
  int **ac = new int*[level+1];          // pointers to A-matrix at each level
  double **av = new double*[level+1];    // and diagonal entries of A-matrix
  double **ad = new double*[level+1];    //

  int **wr = new int*[level];            //
  int **wc = new int*[level];            // pointers to W-matrix at each level
  double **wv = new double*[level];      //

  //initialize first level to original A matrix 
  amg_init(r,c,v,ar,ac,av,ad,matrixSizes,n);

  //Phase 1: AMG setup
  level = amg_setup(ar,ac,av,ad,wr,wc,wv,matrixSizes,level,theta);
  
  //Phase 2: AMG recursive solve
  int ii = 0;
  double err = 1.0;
  while(err>tol && ii<MAX_VCYCLES){
    amg_solve(ar,ac,av,ad,wr,wc,wv,x,b,matrixSizes,n1,n2,level,0);
    #if(FAST_ERROR)
      err = fast_error(ar[0],ac[0],av[0],x,b,n,tol);
    #else
      err = error(ar[0],ac[0],av[0],x,b,n);
    #endif
    #if(DEBUG)
      std::cout<<"error: "<<err<<std::endl;
    #endif
    ii++;
  }
 
  //delete temporary arrays that were allocated on the heap
  for(int i=0;i<level+1;i++){
    delete[] ar[i];
    delete[] ac[i];
    delete[] av[i];
    delete[] ad[i];
  }
  for(int i=0;i<level;i++){   
    delete[] wr[i];
    delete[] wc[i];
    delete[] wv[i];
  }
  delete[] ar;
  delete[] ac;
  delete[] av;
  delete[] ad;
  delete[] wr;
  delete[] wc;
  delete[] wv;
  delete[] matrixSizes;
}




//-------------------------------------------------------------------------------
// AMG recursive solve function
//-------------------------------------------------------------------------------
void amg_solve(int *ar[], int *ac[], double *av[], double *ad[], int *wr[], int *wc[], 
               double *wv[], double x[], const double b[], int ASizes[], int n1, 
               int n2, int level, int currentLevel)
{
  int N = ASizes[currentLevel];     //size of A at each level

  if(currentLevel<level){
    int Nc = ASizes[currentLevel+1];  //size of A at next course level

    //do n1 smoothing steps on A*x=b
    for(int i=0;i<n1;i++){
      //Gauss-Seidel iteration
      double sigma;
      double ajj;
      for(int j=0;j<N;j++){
        sigma = 0.0;
        ajj = ad[currentLevel][j];   //diagonal entry a_jj
        for(int k=ar[currentLevel][j];k<ar[currentLevel][j+1];k++){
          sigma = sigma + av[currentLevel][k]*x[ac[currentLevel][k]];
        }
        x[j] = x[j] + (b[j] - sigma)/ajj;
      }
    }

    //compute residual r=b-A*x=A*e
    double *r = new double[N];
    for(int i=0;i<N;i++){
      double Ax = 0.0;  //matrix vector product Ax
      for(int j=ar[currentLevel][i];j<ar[currentLevel][i+1];j++){
        Ax = Ax + av[currentLevel][j]*x[ac[currentLevel][j]];
      }
      r[i] = b[i] - Ax;
    }   

    //compute W'r 
    double *wres = new double[Nc];
    for(int i=0;i<Nc;i++){wres[i]=0.0;}
    for(int i=0;i<N;i++){
      for(int j=wr[currentLevel][i];j<wr[currentLevel][i+1];j++){
        wres[wc[currentLevel][j]] = wres[wc[currentLevel][j]] + wv[currentLevel][j]*r[i];     
      }
    }

    delete[] r;

    //set e = 0
    double *e = new double[Nc];
    for(int i=0;i<Nc;i++){e[i] = 0.0;}

    //recursizely solve Ac*ec=W'*r
    amg_solve(ar,ac,av,ad,wr,wc,wv,e,wres,ASizes,n1,n2,level,currentLevel+1);
    
    //correct x = x + W*ec
    N = ASizes[currentLevel];   //size of A at each level
    for(int i=0;i<N;i++){
      for(int j=wr[currentLevel][i];j<wr[currentLevel][i+1];j++){
        x[i] = x[i] + wv[currentLevel][j]*e[wc[currentLevel][j]];
      }
    }

    //do n2 smoothing steps on A*x=b
    for(int i=0;i<n2;i++){
      //Gauss-Seidel iteration
      double sigma;
      double ajj;
      for(int j=0;j<N;j++){
        sigma = 0.0;
        ajj = ad[currentLevel][j];  //diagonal entry a_jj
        for(int k=ar[currentLevel][j];k<ar[currentLevel][j+1];k++){
          sigma = sigma + av[currentLevel][k]*x[ac[currentLevel][k]];
        }
        x[j] = x[j] + (b[j] - sigma)/ajj;
      }
    }
  
    delete[] e;
    delete[] wres;
  }
  else{
    //solve A*x=b exactly
    for(int i=0;i<100;i++){
      //Gauss-Seidel iteration
      double sigma;
      double ajj;
      for(int j=0;j<N;j++){
        sigma = 0.0;
        ajj = ad[currentLevel][j];  //diagonal entry a_jj
        for(int k=ar[currentLevel][j];k<ar[currentLevel][j+1];k++){
          sigma = sigma + av[currentLevel][k]*x[ac[currentLevel][k]];
        }
        x[j] = x[j] + (b[j] - sigma)/ajj;
      }
    }
  }
}





//-------------------------------------------------------------------------------
// AMG initialize  function
//-------------------------------------------------------------------------------
void amg_init(const int r[], const int c[], const double v[], int *ar[], int *ac[], 
              double *av[], double *ad[], int ASizes[], const int n)
{
  ASizes[0] = n;
  ar[0] = new int[n+1];
  ac[0] = new int[r[n]];
  av[0] = new double[r[n]];
  ad[0] = new double[n];
  for(int i=0;i<n+1;i++){ar[0][i] = r[i];}
  for(int i=0;i<r[n];i++){ac[0][i] = c[i];}
  for(int i=0;i<r[n];i++){av[0][i] = v[i];}

  //find diagonal entries of A matrix
  for(int i=0;i<n;i++){
    for(int j=r[i];j<r[i+1];j++){
      if(c[j]==i){
        ad[0][i] = v[j];
        break;
      }
    }
  }
}





//-------------------------------------------------------------------------------
// AMG setup phase
//-------------------------------------------------------------------------------
int amg_setup(int *ar[], int *ac[], double *av[], double *ad[], int *wr[], int *wc[], double *wv[], int ASizes[], int level, const double theta)
{
  int rptr_size = 0;

  int i = 0;
  while(ASizes[i]>20 && i<level){
    rptr_size = ASizes[i]+1;  //rptr_size is the size of the row pointer array ar at the current ith level

    //determine size of strength matrix
    int ssize = strength_matrix_size(ar[i],ac[i],av[i],rptr_size,theta);

    //intialize temporary arrays
    int *lambda = new int[rptr_size-1];
    unsigned *cfpoints = new unsigned[rptr_size-1];
    int *srow = new int[rptr_size];
    int *scol = new int[ssize];
    double *sval = new double[ssize];
    int *strow = new int[rptr_size];
    int *stcol = new int[ssize];
    double *stval = new double[ssize];
    for(int j=0;j<rptr_size-1;j++){lambda[j]=0;}
    for(int j=0;j<rptr_size-1;j++){cfpoints[j]=0;}
    for(int j=0;j<rptr_size;j++){srow[j]=0;}
    for(int j=0;j<ssize;j++){scol[j]=0;}
    for(int j=0;j<ssize;j++){sval[j]=0.0;}
    for(int j=0;j<rptr_size;j++){strow[j]=0;}
    for(int j=0;j<ssize;j++){stcol[j]=0;}
    for(int j=0;j<ssize;j++){stval[j]=0.0;}

    //compute strength matrix S
    strength_matrix(ar[i],ac[i],av[i],srow,scol,sval,lambda,rptr_size,theta);

    //compute strength transpose matrix S^T
    strength_transpose_matrix(srow,scol,sval,strow,stcol,stval,lambda,rptr_size,theta);

    //determine c-points and f-points (first pass)
    pre_cpoint3(srow,scol,strow,stcol,lambda,cfpoints,rptr_size);

    //determine c-points and f-points (second pass)
    post_cpoint(srow,scol,cfpoints,rptr_size);

    //compute interpolation matrix W
    int numCPoints = weight_matrix(ar[i],ac[i],av[i],ad[i],srow,scol,sval,wr,wc,wv,cfpoints,rptr_size,i);

    //perform galarkin product Ac = W'*A*W
    galerkin_prod2(ar,ac,av,ad,wr,wc,wv,rptr_size,numCPoints,i);

    ASizes[i+1] = numCPoints;  //size of the A matrix at the next level

    i++;

    //delete temporary arrays
    delete[] cfpoints;
    delete[] lambda;
    delete[] srow;
    delete[] scol;
    delete[] sval;
    delete[] strow;
    delete[] stcol;
    delete[] stval;
  }
  return i;
}




//-------------------------------------------------------------------------------
// function for finding strength matrix size
//-------------------------------------------------------------------------------
int strength_matrix_size(const int r[], const int c[], const double v[], int rptr_size, 
                         const double theta)
{
  int str_size = 0;
  for(int i=0;i<rptr_size-1;i++){
    double max_value = 0.0;
    for(int j=r[i];j<r[i+1];j++){
      if(-v[j]>max_value && i!=c[j]){max_value = fabs(v[j]);}
    }

    max_value = max_value*theta;
    for(int j=r[i];j<r[i+1];j++){
      if(-v[j]>max_value && i!=c[j]){str_size++;}
    }
  }

  return str_size;
}




//-------------------------------------------------------------------------------
// function for finding strength matrix and lambda array
//-------------------------------------------------------------------------------
void strength_matrix(const int r[], const int c[], const double v[], int sr[], int sc[], 
                     double sv[], int lambda[], int rptr_size, const double theta)
{
  //determine strength matrix
  int ind = 0;
  for(int i=0;i<rptr_size-1;i++){
    double max_value = 0.0;
    for(int j=r[i];j<r[i+1];j++){
      if(-v[j]>max_value && i!=c[j]){max_value = fabs(v[j]);}
    }

    max_value = max_value*theta;
    sr[i+1] = sr[i];
    for(int j=r[i];j<r[i+1];j++){
      if(-v[j]>max_value && i!=c[j]){
        sc[ind] = c[j];
        lambda[sc[ind]]++;
        sv[ind] = v[j];
        ind++;
        sr[i+1]++;
      }
    }
  }
}




//-------------------------------------------------------------------------------
// function for finding strength transpose matrix
//-------------------------------------------------------------------------------
void strength_transpose_matrix(int sr[], int sc[], double sv[], int str[], int stc[], 
                               double stv[], int lambda[], int rptr_size, const double theta)
{
  //determine transpose strength matrix
  for(int i=1;i<rptr_size;i++){str[i]=lambda[i-1]+str[i-1];}

  unsigned *tmp = new unsigned[rptr_size-1];
  for(int i=0;i<rptr_size-1;i++){tmp[i] = 0;}
  for(int i=0;i<rptr_size-1;i++){
    for(int j=sr[i];j<sr[i+1];j++){
      stc[str[sc[j]]+tmp[sc[j]]] = i;
      stv[str[sc[j]]+tmp[sc[j]]] = sv[j];
      tmp[sc[j]]++;
    }
  }
  
  delete[] tmp;

  //DEBUG
  //cout<<""<<endl; cout<<"sr"<<endl;
  //for(int i=0;i<rptr_size;i++){
  //  cout<<str[i]<<" ";
  //}
  //cout<<""<<endl; cout<<"sc"<<endl;
  //for(int i=0;i<str[rptr_size-1];i++){
  //  cout<<stc[i]<<" ";
  //}
  //cout<<""<<endl; cout<<"sv"<<endl;
  //for(int i=0;i<str[rptr_size-1];i++){
  //  cout<<stv[i]<<" ";
  //}
  //cout<<""<<endl;
}



 
//-------------------------------------------------------------------------------
// function for finding c-points and f-points (first pass)
//-------------------------------------------------------------------------------
void pre_cpoint(int sr[], int sc[], int str[], int stc[], int lambda[], unsigned cfpoints[], 
                int rptr_size)
{
  unsigned idSize = rptr_size-1;
  unsigned *id = new unsigned[idSize];
  for(unsigned i=0;i<idSize;i++){id[i] = i;}

  int num_nodes_not_assign = rptr_size-1;
  while(num_nodes_not_assign>0)
  {
    int max_value = -999;
    unsigned max_index = 0;
    unsigned index = 0;
    for(unsigned int i=0;i<idSize;i++){
      if(lambda[id[i]]!=-999){
        if(lambda[id[i]]>max_value){
          max_value = lambda[id[i]];
          max_index = id[i];
        }
        id[index] = id[i];
        index++;
      }
    }
    idSize = index;

    cfpoints[max_index] = 1;
    lambda[max_index] = -999;
    num_nodes_not_assign--;

    //determine how many nonzero entries are in the max_index column of S and 
    //what rows those nonzero values are in
    int count = 0;
    int nnz_in_col = str[max_index+1]-str[max_index]; 
    int *index_of_nz = new int[nnz_in_col];
    for(int i=str[max_index];i<str[max_index+1];i++){
      index_of_nz[i-str[max_index]] = stc[i];
    }

    //make all connections to cpoint fpoints and update lambda array
    for(int i=0;i<nnz_in_col;i++){
      if(lambda[index_of_nz[i]]!=-999){
        lambda[index_of_nz[i]] = -999;
        num_nodes_not_assign--;
        for(int j=sr[index_of_nz[i]];j<sr[index_of_nz[i]+1];j++){
          if(lambda[sc[j]]!=-999){lambda[sc[j]]++;}
        }
      }
    }
    delete[] index_of_nz;
  }
  delete[] id;

  //DEBUG
  //cout<<"cfpoints: "<<" "<<endl;
  //for(int i=0;i<rptr_size-1;i++){cout<<cfpoints[i]<<"  ";}
  //cout<<""<<endl;
}





//-------------------------------------------------------------------------------
// function for finding c-points and f-points (first pass)
//-------------------------------------------------------------------------------
void pre_cpoint3(int sr[], int sc[], int str[], int stc[], int lambda[], unsigned cfpoints[], 
                 int rptr_size)
{
  unsigned locInSortedLambda = 0;
  unsigned numOfNodesToCheck = 0;
  unsigned *nodesToCheck = new unsigned[rptr_size-1];
  struct array *sortedLambda = new array[rptr_size-1];
  for(int i=0;i<rptr_size-1;i++){nodesToCheck[i] = 0;}

  //copy lambda into struct array and then sort
  for(int i=0;i<rptr_size-1;i++){
    sortedLambda[i].value = lambda[i];
    sortedLambda[i].id = i;
  }
  qsort(sortedLambda, rptr_size-1, sizeof(sortedLambda[0]), compare_structs);
  nodesToCheck[0] = sortedLambda[0].id;
  numOfNodesToCheck++;

  int num_nodes_not_assign = rptr_size-1;
  while(num_nodes_not_assign>0)
  {
    int max_value = -999;
    unsigned max_index = 0;
    while(locInSortedLambda<rptr_size-2 && lambda[sortedLambda[locInSortedLambda].id]==-999){
      locInSortedLambda++;
    }
    nodesToCheck[0] = sortedLambda[locInSortedLambda].id;

    for(unsigned i=0;i<numOfNodesToCheck;i++){
      if(lambda[nodesToCheck[i]]>max_value){
        max_value = lambda[nodesToCheck[i]];
        max_index = nodesToCheck[i];
      }
    }
    numOfNodesToCheck = 1;

    cfpoints[max_index] = 1;
    lambda[max_index] = -999;
    num_nodes_not_assign--;

    //determine how many nonzero entries are in the max_index column of S and 
    //what rows those nonzero values are in
    int nnz_in_col = str[max_index+1]-str[max_index];
    int *index_of_nz = new int[nnz_in_col];
    for(int i=0;i<nnz_in_col;i++){index_of_nz[i] = stc[i+str[max_index]];}

    //make all connections to cpoint fpoints and update lambda array
    for(int i=0;i<nnz_in_col;i++){
      if(lambda[index_of_nz[i]]!=-999){
        lambda[index_of_nz[i]] = -999;
        num_nodes_not_assign--;
        for(int j=sr[index_of_nz[i]];j<sr[index_of_nz[i]+1];j++){
          if(lambda[sc[j]]!=-999){
            lambda[sc[j]]++;
            int flag = 0;
            for(unsigned k=0;k<numOfNodesToCheck;k++){
              if(nodesToCheck[k]==sc[j]){
                flag = 1;
                break;
              }
            }
            if(flag==0){
              nodesToCheck[numOfNodesToCheck] = sc[j];
              numOfNodesToCheck++;
            }
          }
        }
      }
    }
    delete[] index_of_nz;
  }
  delete[] nodesToCheck;
  delete[] sortedLambda;
}




//-------------------------------------------------------------------------------
// function for finding c-points and f-points (second pass)
//-------------------------------------------------------------------------------
void post_cpoint(int sr[], int sc[], unsigned cfpoints[], int rptr_size)
{
  int max_nstrc = 0;  //max number of strong connections in any row
  for(int i=0;i<rptr_size-1;i++){
    if(max_nstrc<sr[i+1]-sr[i]){max_nstrc = sr[i+1]-sr[i];}
  }
 
  int *scpoints = new int[max_nstrc]; 

  //perform second pass adding c-points where necessary
  for(int i=0;i<rptr_size-1;i++){
    if(cfpoints[i]==0){                //i is an fpoint
      int nstrc = sr[i+1]-sr[i];       //number of strong connections in row i
      int scindex = 0;                 //number of c-points in row i
      for(int j=sr[i];j<sr[i+1];j++){
        if(cfpoints[sc[j]]==1){
          scpoints[scindex] = sc[j];
          scindex++;
        }
      }

      #if(DEBUG)
        if(scindex==0){std::cout<<"ERROR: no cpoint for the f-point "<<i<<std::endl;}
      #endif

      for(int j=sr[i];j<sr[i+1];j++){
        if(cfpoints[sc[j]]==0){  //sc[j] is an fpoint
          int ind1 = 0, ind2 = 0, flag = 1;
          while(ind1<scindex && ind2<(sr[sc[j]+1]-sr[sc[j]])){
            if(scpoints[ind1]==sc[sr[sc[j]]+ind2]){
              flag = 0;
              break;
            }
            else if(scpoints[ind1]<sc[sr[sc[j]]+ind2]){
              ind1++;
            }
            else if(scpoints[ind1]>sc[sr[sc[j]]+ind2]){
              ind2++;
            }
          }
          if(flag){
            cfpoints[sc[j]] = 1; // sc[j] was an fpoint, but now is a cpoint 
            scpoints[scindex] = sc[j];
            scindex++;
          }
        }
      }
    }
  }

  delete[] scpoints;

  //DEBUG
  //cout<<"cfpoints: "<<" "<<endl;
  //for(int i=0;i<rptr_size-1;i++){cout<<cfpoints[i]<<"  ";}
  //cout<<""<<endl;
}




//-------------------------------------------------------------------------------
// function for finding interpolation weight matrix
//-------------------------------------------------------------------------------
int weight_matrix(const int r[], const int c[], const double v[], double d[], int sr[], 
                  int sc[], double sv[], int *wr[], int *wc[], double *wv[], 
                  unsigned cfpoints[], int rptr_size, int level)
{
  //determine the number of c-points and f-points
  int cnum = 0;
  int fnum = 0;
  for(int i=0;i<rptr_size-1;i++){cnum += cfpoints[i];}
  fnum = rptr_size-1-cnum;

  //determine the size of the interpolation matrix W
  int wsize=cnum;
  for(int i=0;i<rptr_size-1;i++){
    if(cfpoints[i]==0){
      for(int j=sr[i];j<sr[i+1];j++){
        if(cfpoints[sc[j]]==1){wsize++;}
      }
    }
  }

  //initialize interpolation matrix W
  wr[level] = new int[rptr_size];
  wc[level] = new int[wsize];
  wv[level] = new double[wsize];
  for(int j=0;j<rptr_size;j++){wr[level][j]=0;}
  for(int j=0;j<wsize;j++){wc[level][j]=-1;}
  for(int j=0;j<wsize;j++){wv[level][j]=0.0;}

  //modify cfpoints array so that nonzeros now correspond to the cpoint location
  int loc = 0;
  for(int i=0;i<rptr_size-1;i++){
    if(cfpoints[i]==1){
      cfpoints[i] = cfpoints[i] + loc;
      loc++;
    }
  } 

  //find beta array (sum of weak f-points)
  int ind1 = 0, ind2 = 0, ii = 0;
  double *beta = new double[fnum];
  for(int i=0;i<fnum;i++){beta[i] = 0.0;}
  for(int i=0;i<rptr_size-1;i++){
    if(cfpoints[i]==0){
      ind1 = 0;
      ind2 = 0;
      while(ind1<(r[i+1]-r[i]) && ind2<(sr[i+1]-sr[i])){
        if(c[r[i]+ind1]==sc[sr[i]+ind2]){
          ind1++;
          ind2++;
        }
        else if(c[r[i]+ind1]<sc[sr[i]+ind2]){
          if(c[r[i]+ind1]!=i){
            beta[ii] = beta[ii] + v[r[i]+ind1];
          }
          ind1++;
        }
      }
      while(ind1<(r[i+1]-r[i])){
        if(c[r[i]+ind1]!=i){
          beta[ii] = beta[ii] + v[r[i]+ind1];
        }
        ind1++;
      }
      ii++;
    }
  }  

  //create interpolation matrix W
  double aii = 0.0, aij = 0.0, temp = 0.0;
  int index = 0, rindex = 0;
  ind1 = 0;
  ind2 = 0;
  for(int i=0;i<rptr_size-1;i++){
    if(cfpoints[i]>=1){
      wc[level][index] = ind1;
      wv[level][index] = 1.0;
      ind1++;
      index++;
      rindex++;
      wr[level][rindex] = wr[level][rindex-1] + 1;
    }
    else{
      //determine diagonal element a_ii
      aii = d[i];
  
      //find all strong c-points and f-points in the row i
      int ind3 = 0, ind4 = 0;
      int scnum = 0;
      int sfnum = 0;
      int *scpts = new int[sr[i+1]-sr[i]];
      int *sfpts = new int[sr[i+1]-sr[i]];
      int *scind = new int[sr[i+1]-sr[i]];
      double *scval = new double[sr[i+1]-sr[i]];
      double *sfval = new double[sr[i+1]-sr[i]];
      for(int j=0;j<(sr[i+1]-sr[i]);j++){
        scpts[j] = -1;
        sfpts[j] = -1;
        scind[j] = -1;
        scval[j] = 0.0;
        sfval[j] = 0.0;
      }
      for(int j=sr[i];j<sr[i+1];j++){
        if(cfpoints[sc[j]]>=1){
          scpts[scnum] = sc[j];
          scval[scnum] = sv[j];
          scind[scnum] = cfpoints[sc[j]]-1;
          scnum++;
        }
        else{
          sfpts[sfnum] = sc[j];
          sfval[sfnum] = sv[j];
          sfnum++;
        }
      }

      #if(DEBUG)
        if(scnum==0){std::cout<<"ERROR: no cpoints in row "<<i<<std::endl;}
      #endif
   
      if(sfnum==0){
        //loop all strong c-points 
        for(int k=0;k<scnum;k++){
          aij = scval[k];
          wc[level][index] = scind[k];
          wv[level][index] = -(aij)/(aii + beta[ind2]);
          index++;
        }
      }
      else{
        //loop thru all the strong f-points to find alpha array
        double *alpha = new double[sfnum];
        for(int k=0;k<sfnum;k++){alpha[k] = 0.0;}
        for(int k=0;k<sfnum;k++){
          ind3 = 0;
          ind4 = 0;
          while(ind3<scnum && ind4<(r[sfpts[k]+1]-r[sfpts[k]])){
            if(scpts[ind3]==c[r[sfpts[k]]+ind4]){
              alpha[k] = alpha[k] + v[r[sfpts[k]]+ind4];
              ind3++;
              ind4++; 
            }
            else if(scpts[ind3]<c[r[sfpts[k]]+ind4]){
              ind3++;
            }
            else if(scpts[ind3]>c[r[sfpts[k]]+ind4]){
              ind4++;
            }
          }
        }

        //loop all strong c-points 
        for(int k=0;k<scnum;k++){
          aij = scval[k];
          temp = 0.0;
          for(int l=0;l<sfnum;l++){
            for(int m=r[sfpts[l]];m<r[sfpts[l]+1];m++){
              if(c[m]==scpts[k]){
                #if(DEBUG)
                  if(alpha[l]==0.0){std::cout<<"ERROR: alpha is zero"<<std::endl;}
                #endif
                temp = temp + sfval[l]*v[m]/alpha[l];
                break;
              } 
            }
          }
          wc[level][index] = scind[k];
          wv[level][index] = -(aij + temp)/(aii + beta[ind2]);
          index++;
        }

        delete[] alpha;
      }
      ind2++;
      rindex++;
      wr[level][rindex] = wr[level][rindex-1] + scnum;

      delete[] scpts;
      delete[] sfpts;
      delete[] scind;
      delete[] scval;
      delete[] sfval;
    }
  }

  delete[] beta;

  //DEBUG
  //cout<<""<<endl;
  //for(int i=0;i<rptr_size;i++){
  //  cout<<wr[level][i]<<" "<<endl;
  //}
  //cout<<""<<endl;
  //for(int i=0;i<wsize;i++){
  //  cout<<wc[level][i]<<" "<<endl;
  //}
  //cout<<""<<endl;
  //for(int i=0;i<wsize;i++){
  //  cout<<wv[level][i]<<" "<<endl;
  //}
  //cout<<""<<endl;
 
  return cnum;
}




//-------------------------------------------------------------------------------
// function for performing galarkin product: W'*A*W
//-------------------------------------------------------------------------------
void galerkin_prod(int *ar[], int *ac[], double *av[], int *wr[], int *wc[], double *wv[], 
                   int rptr_size, int m, int level)
{
  int n = rptr_size-1; //number of rows in A and W. 
  int *nnzIthWCol = new int[m];
  int *nnzJthARow = new int[n];
  int *nnzIthAWRow = new int[n];
  int *nnzIthWAWRow = new int[m];

  int *temp1 = new int[m];
  double *temp2 = new double[m];

  int *wpr = new int[m+1];
  int *wpc = new int[wr[level][n]];
  double *wpv = new double[wr[level][n]];

  //initialize nnzIthWCol, nnzJthARow, nnzIthAWRow, and nnzIthWAWRow to zero
  for(int i=0;i<m;i++){nnzIthWCol[i]=0;}
  for(int i=0;i<n;i++){nnzJthARow[i]=0;}
  for(int i=0;i<n;i++){nnzIthAWRow[i]=0;}
  for(int i=0;i<m;i++){nnzIthWAWRow[i]=0;}

  //first determine how many non-zeros exist in each column of W and store in array nnzIthWCol
  for(int i=0;i<wr[level][n];i++){nnzIthWCol[wc[level][i]] = nnzIthWCol[wc[level][i]] + 1;}

  //second determine how many non-zeros exist in each row of A and store in array nnzJthARow
  for(int i=0;i<n;i++){nnzJthARow[i] = ar[level][i+1]-ar[level][i];}

  //find W' in CRS format from W
  temp1[0] = 0; wpr[0] = 0;
  for(int i=1;i<m;i++){temp1[i] = temp1[i-1] + nnzIthWCol[i-1];}
  for(int i=1;i<m+1;i++){wpr[i] = wpr[i-1] + nnzIthWCol[i-1];}
  for(int i=0;i<n;i++){
    for(int j=wr[level][i];j<wr[level][i+1];j++){
      wpc[temp1[wc[level][j]]] = i;
      wpv[temp1[wc[level][j]]] = wv[level][j];
      temp1[wc[level][j]]++;
    }
  }

  //Now determine how many non-zeros exist in the matrix product of A*W
  int nnz = 0;
  for(int i=0;i<m;i++){temp1[i] = 0;}
  for(int i=0;i<n;i++){ //loop through each row of A
    for(int j=ar[level][i];j<ar[level][i+1];j++){
      for(int k=wr[level][ac[level][j]];k<wr[level][ac[level][j]+1];k++){
        if(temp1[wc[level][k]]!=-(i+1)){
          nnz++;
          temp1[wc[level][k]]=-(i+1);
        }
      }
    }
  }

  //create temporary arrays for storing the result of A*W and initialize to zeros
  int *tr = new int[n+1];
  int *tc = new int[nnz];
  double *tv = new double[nnz];
  for(int i=0;i<nnz;i++){
    tc[i] = 0;
    tv[i] = 0.0;
  }

  //now compute the matrix product A*W and store the result in tc and tv
  int indx = 0;
  for(int i=0;i<n;i++){ //loop through each row of A
    for(int j=0;j<m;j++){temp1[j] = -1;}
    for(int j=0;j<m;j++){temp2[j] = 0.0;}
    for(int j=ar[level][i];j<ar[level][i+1];j++){
      for(int k=wr[level][ac[level][j]];k<wr[level][ac[level][j]+1];k++){
        if(temp1[wc[level][k]]==-1){  //new column 
          temp1[wc[level][k]] = wc[level][k];
          temp2[wc[level][k]] = temp2[wc[level][k]] + av[level][j]*wv[level][k];
        }
        else{
          temp2[wc[level][k]] = temp2[wc[level][k]] + av[level][j]*wv[level][k];
        }
      }
    }
    for(int l=0;l<m;l++){
      if(temp1[l]!=-1){
        tc[indx] = temp1[l];
        tv[indx] = tv[indx] + temp2[l];
        nnzIthAWRow[i]++;
        indx++;
      }
    }
  }

  //update tc pointer array
  tr[0] = 0;
  for(int i=1;i<n+1;i++){tr[i] = tr[i-1] + nnzIthAWRow[i-1];}

  //We know have the produxt T=A*W given by the arrays: tr, tc, & tv. Need to perform the product B=W'T 
  //Now determine how many non-zeros exist in the matrix product of W'T
  nnz = 0;
  for(int i=0;i<m;i++){temp1[i] = 0;}
  for(int i=0;i<m;i++){ //loop through each row of W'
    for(int j=wpr[i];j<wpr[i+1];j++){
      for(int k=tr[wpc[j]];k<tr[wpc[j]+1];k++){
        if(temp1[tc[k]]!=-(i+1)){
          nnz++;
          temp1[tc[k]]=-(i+1);
        }
      }
    }
  }
 
  //create arrays for storing the result of W'*A*W and initialize to zeros
  ar[level+1] = new int[m+1];
  ac[level+1] = new int[nnz];
  av[level+1] = new double[nnz];

  for(int i=0;i<m+1;i++){ar[level+1][i] = 0;}
  for(int i=0;i<nnz;i++){
    ac[level+1][i] = 0;
    av[level+1][i] = 0.0;
  }

  //now compute the matrix product W'T and store the result in ac and av
  indx = 0;
  for(int i=0;i<m;i++){ //loop through each row of W'
    for(int j=0;j<m;j++){temp1[j] = -1;}
    for(int j=0;j<m;j++){temp2[j] = 0.0;}
    for(int j=wpr[i];j<wpr[i+1];j++){
      for(int k=tr[wpc[j]];k<tr[wpc[j]+1];k++){
        if(temp1[tc[k]]==-1){  //new column 
          temp1[tc[k]] = tc[k];
          temp2[tc[k]] = temp2[tc[k]] + wpv[j]*tv[k];
        }
        else{
          temp2[tc[k]] = temp2[tc[k]] + wpv[j]*tv[k];
        }
      }
    }
    for(int l=0;l<m;l++){
      if(temp1[l]!=-1){
        ac[level+1][indx] = temp1[l];
        av[level+1][indx] = av[level+1][indx] + temp2[l];
        nnzIthWAWRow[i]++;
        indx++;
      }
    }
  }

  //update ar pointer array
  ar[level+1][0] = 0;
  for(int i=1;i<m+1;i++){ar[level+1][i] = ar[level+1][i-1] + nnzIthWAWRow[i-1];}

  delete[] nnzIthWCol;
  delete[] nnzJthARow;
  delete[] nnzIthAWRow;
  delete[] nnzIthWAWRow;
  delete[] temp1;
  delete[] temp2;
  delete[] wpr;
  delete[] wpc;
  delete[] wpv;
  delete[] tr;
  delete[] tc;
  delete[] tv;

  //DEBUG
  //cout<<"ar: "<<endl;
  //for(int i=0;i<m+1;i++){cout<<ar[level+1][i]<<endl;}
  //cout<<"ac: "<<endl;
  //for(int i=0;i<nnz;i++){cout<<ac[level+1][i]<<endl;}
  //cout<<"av: "<<endl;
  //for(int i=0;i<nnz;i++){cout<<av[level+1][i]<<endl;}
}




//-------------------------------------------------------------------------------
// function for performing galarkin product: W'*A*W
//-------------------------------------------------------------------------------
void galerkin_prod2(int *ar[], int *ac[], double *av[], double *ad[], int *wr[], 
                    int *wc[], double *wv[], int rptr_size, int m, int level)
{
  int n = rptr_size-1; //number of rows in A and W. 
  //int *nnzIthWCol = new int[m];

  int *temp1 = new int[m];

  int *wpr = new int[m+1];
  int *wpc = new int[wr[level][n]];
  double *wpv = new double[wr[level][n]];

  //initialize nnzIthWCol to zero
  //for(int i=0;i<m;i++){nnzIthWCol[i]=0;}
  for(int i=0;i<m;i++){temp1[i]=0;}

  //first determine how many non-zeros exist in each column of W and store in array nnzIthWCol
  //for(int i=0;i<wr[level][n];i++){nnzIthWCol[wc[level][i]] = nnzIthWCol[wc[level][i]] + 1;}
  for(int i=0;i<wr[level][n];i++){temp1[wc[level][i]]++;}

  //find W' in CRS format from W
  wpr[0] = 0;
  //for(int i=1;i<m+1;i++){wpr[i] = wpr[i-1] + nnzIthWCol[i-1];}
  for(int i=1;i<m+1;i++){wpr[i] = wpr[i-1] + temp1[i-1];}
  //for(int i=1;i<m;i++){temp1[i] = temp1[i-1] + nnzIthWCol[i-1];}
  temp1[0] = 0;
  for(int i=1;i<m;i++){temp1[i] = temp1[i-1] + wpr[i] - wpr[i-1];}
  for(int i=0;i<n;i++){
    for(int j=wr[level][i];j<wr[level][i+1];j++){
      wpc[temp1[wc[level][j]]] = i;
      wpv[temp1[wc[level][j]]] = wv[level][j];
      temp1[wc[level][j]]++;
    }
  }

  //Now determine how many non-zeros exist in the matrix product of A*W
  int nnz = 0;
  for(int i=0;i<m;i++){temp1[i] = 0;}
  for(int i=0;i<n;i++){ //loop through each row of A
    for(int j=ar[level][i];j<ar[level][i+1];j++){
      for(int k=wr[level][ac[level][j]];k<wr[level][ac[level][j]+1];k++){
        if(temp1[wc[level][k]]!=-(i+1)){
          nnz++;
          temp1[wc[level][k]]=-(i+1);
        }
      }
    }
  }

  //create temporary arrays for storing the result of A*W and initialize to zeros
  int *tr = new int[n+1];
  int *tc = new int[nnz];
  double *tv = new double[nnz];
  for(int i=0;i<nnz;i++){
    tc[i] = 0;
    tv[i] = 0.0;
  }

  //now compute the matrix product A*W and store the result in tr, tc and tv
  tr[0] = 0;
  int indx = 0, start = 0;
  for(int i=0;i<n;i++){ //loop through each row of A
    for(int j=ar[level][i];j<ar[level][i+1];j++){
      for(int k=wr[level][ac[level][j]];k<wr[level][ac[level][j]+1];k++){
        int found = 0;
        for(int l=start;l<indx;l++){
          if(tc[l]==wc[level][k]){
            tv[l] = tv[l] + av[level][j]*wv[level][k];
            found = 1;
            break;
          }
        }
        if(found==0){
          tc[indx] = wc[level][k];
          tv[indx] = av[level][j]*wv[level][k];
          indx++;
        }
      }
    }
    //sort tc[start:indx-1]
    sort(tc,tv,start,indx);
    start = indx;
    tr[i+1] = indx;
  }

  //We know have the produxt T=A*W given by the arrays: tr, tc, & tv. Need to perform the 
  //product B=W'T. Now determine how many non-zeros exist in the matrix product of W'T
  nnz = 0;
  for(int i=0;i<m;i++){temp1[i] = 0;}
  for(int i=0;i<m;i++){ //loop through each row of W'
    for(int j=wpr[i];j<wpr[i+1];j++){
      for(int k=tr[wpc[j]];k<tr[wpc[j]+1];k++){
        if(temp1[tc[k]]!=-(i+1)){
          nnz++;
          temp1[tc[k]]=-(i+1);
        }
      }
    }
  }

  //create arrays for storing the result of W'*A*W and initialize to zeros
  ar[level+1] = new int[m+1];
  ac[level+1] = new int[nnz];
  av[level+1] = new double[nnz];
  ad[level+1] = new double[m];
  for(int i=0;i<nnz;i++){
    ac[level+1][i] = 0;
    av[level+1][i] = 0.0;
  }

  //now compute the matrix product W'T and store the result in ac and av
  ar[level+1][0] = 0;
  indx = 0, start = 0;
  for(int i=0;i<m;i++){ //loop through each row of W'
    for(int j=wpr[i];j<wpr[i+1];j++){
      for(int k=tr[wpc[j]];k<tr[wpc[j]+1];k++){
        int found = 0;
        for(int l=start;l<indx;l++){
          if(ac[level+1][l]==tc[k]){
            av[level+1][l] = av[level+1][l] + wpv[j]*tv[k];
            found = 1;
            break;
          }
        }
        if(found==0){
          ac[level+1][indx] = tc[k];
          av[level+1][indx] = wpv[j]*tv[k];
          indx++;
        }
      }
    }
    //sort tc[start:indx-1]
    sort(ac[level+1],av[level+1],start,indx);
    start = indx;
    ar[level+1][i+1] = indx;
  }

  //find diagonal entries of A matrix
  for(int i=0;i<m;i++){
    for(int j=ar[level+1][i];j<ar[level+1][i+1];j++){
      if(ac[level+1][j]==i){
        ad[level+1][i] = av[level+1][j];
        break;
      }
    } 
  }

  //delete[] nnzIthWCol;
  delete[] temp1;
  delete[] wpr;
  delete[] wpc;
  delete[] wpv;
  delete[] tr;
  delete[] tc;
  delete[] tv;

  //DEBUG
  //cout<<"ar: "<<endl;
  //for(int i=0;i<m+1;i++){cout<<ar[level+1][i]<<"  ";}
  //cout<<"ac: "<<endl;
  //for(int i=0;i<nnz;i++){cout<<ac[level+1][i]<<endl;}
  //cout<<"av: "<<endl;
  //for(int i=0;i<nnz;i++){cout<<av[level+1][i]<<endl;}
  //std::cout<<"ad: "<<std::endl;
  //for(int i=0;i<m;i++){std::cout<<ad[level+1][i]<<std::endl;}
}





//-------------------------------------------------------------------------------
// sort function used in galarkin2
//-------------------------------------------------------------------------------
inline void sort(int array1[], double array2[], int start, int end)
{
  for(int i=start;i<end;i++){
    int index = i;
    for(int j=i+1;j<end;j++){
      if(array1[index]>array1[j]){
        index = j;
      }
    }
    int temp1 = array1[i];
    double temp2 = array2[i];
    array1[i] = array1[index];
    array2[i] = array2[index];
    array1[index] = temp1;
    array2[index] = temp2;
  }
}



//-------------------------------------------------------------------------------
// compare function for sorting structure array
//-------------------------------------------------------------------------------
int compare_structs(const void *a, const void *b){
    array *struct_a = (array *) a;
    array *struct_b = (array *) b;

    if (struct_a->value < struct_b->value) return 1;
    else if (struct_a->value == struct_b->value) return 0;
    else return -1;
}



















































































//-------------------------------------------------------------------------------
// function for performing galarkin product: W'*A*W
//-------------------------------------------------------------------------------
void galerkin_prod3(int *ar[], int *ac[], double *av[], int *wr[], int *wc[], double *wv[], 
                    int rptr_size, int m, int level)
{
  int n = rptr_size-1; //number of rows in A and W. 
  int *nnzIthWCol = new int[m];

  int *temp1 = new int[m];
  int *temp2 = new int[n];

  int *wpr = new int[m+1];
  int *wpc = new int[wr[level][n]];
  double *wpv = new double[wr[level][n]];

  //initialize nnzIthWCol to zero
  for(int i=0;i<m;i++){nnzIthWCol[i]=0;}

  //first determine how many non-zeros exist in each column of W and store in array nnzIthWCol
  for(int i=0;i<wr[level][n];i++){nnzIthWCol[wc[level][i]] = nnzIthWCol[wc[level][i]] + 1;}

  //find W' in CRS format from W
  temp1[0] = 0; wpr[0] = 0;
  for(int i=1;i<m;i++){temp1[i] = temp1[i-1] + nnzIthWCol[i-1];}
  for(int i=1;i<m+1;i++){wpr[i] = wpr[i-1] + nnzIthWCol[i-1];}
  for(int i=0;i<n;i++){
    for(int j=wr[level][i];j<wr[level][i+1];j++){
      wpc[temp1[wc[level][j]]] = i;
      wpv[temp1[wc[level][j]]] = wv[level][j];
      temp1[wc[level][j]]++;
    }
  }


  //Now determine how many non-zeros exist in the matrix product of W'A
  int nnz = 0;
  for(int i=0;i<n;i++){temp2[i] = 0;}
  for(int i=0;i<m;i++){ //loop through each row of W'
    for(int j=wpr[i];j<wpr[i+1];j++){
      for(int k=ar[level][wpc[j]];k<ar[level][wpc[j]+1];k++){
        if(temp2[ac[level][k]]!=-(i+1)){
          nnz++;
          temp2[ac[level][k]]=-(i+1);
        }
      }
    }
  }

  //cout<<"nnz: "<<nnz<<endl;
  //cout<<"n: "<<n<<endl;
  //cout<<"m: "<<m<<endl;

  //create temporary arrays for storing the result of W'*A and initialize to zeros
  int *tr = new int[m+1];
  int *tc = new int[nnz];
  double *tv = new double[nnz];
  for(int i=0;i<m+1;i++){tr[i] = 0;}
  for(int i=0;i<nnz;i++){
    tc[i] = 0;
    tv[i] = 0.0;
  }

  //now compute the matrix product W'*A and store the result in tr, tc, and tv
  int indx = 0;
  for(int i=0;i<m;i++){  //loop through rows of W'
    for(int j=0;j<n;j++){  //loop through cols of A (use fact that A is symmetric)
      int ind1 = 0, ind2 = 0, found = 0;
      while(ind1<(wpr[i+1]-wpr[i]) && ind2<(ar[level][j+1]-ar[level][j])){
        if(wpc[ind1+wpr[i]]==ac[level][ind2+ar[level][j]]){
          tc[indx] = j;
          tv[indx] = tv[indx] + av[level][ind2+ar[level][j]]*wpv[ind1+wpr[i]];
          ind1++;
          ind2++;
          found = 1;
        }
        else if(wpc[ind1+wpr[i]]<ac[level][ind2+ar[level][j]]){
          ind1++;
        }
        else if(wpc[ind1+wpr[i]]>ac[level][ind2+ar[level][j]]){
          ind2++;
        }
      }
      if(found==1){indx++;}
    }
    //cout<<"indx: "<<indx<<endl;
    tr[i+1] = indx;
  }

  //for(int i=0;i<m+1;i++){cout<<tr[i]<<"  ";}

  //We now have the produxt T=W'*A given by the arrays: tr, tc, & tv. Need to perform the product B=T*W 
  //Now determine how many non-zeros exist in the matrix product of T*W
  nnz = 0;
  for(int i=0;i<m;i++){  //loop through rows of T
    for(int j=0;j<m;j++){  //loop through cols of W
      int ind1 = 0, ind2 = 0, found = 0;
      while(ind1<(tr[i+1]-tr[i]) && ind2<(wpr[j+1]-wpr[j])){
        if(tc[ind1+tr[i]]==wpc[ind2+wpr[j]]){
          nnz++;
          break;
        }
        else if(tc[ind1+tr[i]]<wpc[ind2+wpr[j]]){
          ind1++;
        }
        else if(tc[ind1+tr[i]]>wpc[ind2+wpr[j]]){
          ind2++;
        }
      }
    }
  }


  //for(int i=0;i<n;i++){temp2[i] = 0;}
  //for(int i=0;i<m;i++){ //loop through each row of T
  //  for(int j=tr[i];j<tr[i+1];j++){
  //    for(int k=wpr[tc[j]];k<wpr[tc[j]+1];k++){
  //      if(temp2[wpc[k]]!=-(i+1)){
  //        nnz++;
  //        temp2[wpc[k]]=-(i+1);
  //      }
  //    }
  //  }
  //}

  //create arrays for storing the result of W'*A*W and initialize to zeros
  ar[level+1] = new int[m+1];
  ac[level+1] = new int[nnz];
  av[level+1] = new double[nnz];

  for(int i=0;i<m+1;i++){ar[level+1][i] = 0;}
  for(int i=0;i<nnz;i++){
    ac[level+1][i] = 0;
    av[level+1][i] = 0.0;
  }

  //now compute the matrix product T*W and store the result in ar, ac, and av
  indx = 0;
  for(int i=0;i<m;i++){  //loop through rows of T
    for(int j=0;j<m;j++){  //loop through cols of W 
      int ind1 = 0, ind2 = 0, found = 0;
      while(ind1<(tr[i+1]-tr[i]) && ind2<(wpr[j+1]-wpr[j])){
        if(tc[ind1+tr[i]]==wpc[ind2+wpr[j]]){
          ac[level+1][indx] = j;
          av[level+1][indx] = av[level+1][indx] + tv[ind1+tr[i]]*wpv[ind2+wpr[j]];
          ind1++;
          ind2++;
          found = 1;
        }
        else if(tc[ind1+tr[i]]<wpc[ind2+wpr[j]]){
          ind1++;
        }
        else if(tc[ind1+tr[i]]>wpc[ind2+wpr[j]]){
          ind2++;
        }
      }
      if(found==1){indx++;}
    }
    ar[level+1][i+1] = indx;
  }

  delete[] nnzIthWCol;
  delete[] temp1;
  delete[] temp2;
  delete[] wpr;
  delete[] wpc;
  delete[] wpv;
  delete[] tr;
  delete[] tc;
  delete[] tv;
}




//#if(DEBUG)
//  if((aii + beta[ind2])==0.0){
//    std::cout<<"ERROR: "<<-(aij + temp)/(aii + beta[ind2])<<std::endl;
//    std::cout<<"row: "<<i<<std::endl;
//    std::cout<<(aii + beta[ind2])<<std::endl;
//    std::cout<<"aii: "<<aii<<std::endl;
//    std::cout<<"ind2: "<<ind2<<std::endl;
//    std::cout<<"beta[ind2]: "<<beta[ind2]<<std::endl;
//  }
//#endif
