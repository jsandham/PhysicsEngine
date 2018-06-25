#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <time.h>

#include "debug.h"
#include "AMG.h"
//#include "PCG.h"
//#include "RICH.h"
//#include "JGS.h"
using namespace std;


int DEBUG_TEST_MATRIX_AMG(std::string filename, unsigned int nnz, unsigned int nr, unsigned int solver)
{
	ifstream fin(filename.c_str());         
	int N = nnz; 
	int Nr = nr;


	int currentLine = 0;
  	int index = 0;
  	int *rows = new int[N];
  	int *columns = new int[N];
  	double *values = new double[N];
  	for(int i=0;i<N;i++){
    	rows[i]=0;
    	columns[i]=0;
    	values[i]=0.0;
  	}
  	string percent("%");
  	string space(" ");
  	string token;

  	//scan through file
  	while(!fin.eof())
  	{
    	string line;
    	getline(fin,line);

    	if(index==N){break;}

    	// parse the line
    	if(line.substr(0,1).compare(percent)!=0){
      		if(currentLine>0){
        		token = line.substr(0,line.find(space));
        		columns[index] = atoi(token.c_str())-1;
        		line.erase(0,line.find(space)+space.length());
		        token = line.substr(0,line.find(space));
		        rows[index] = atoi(token.c_str())-1;
		        line.erase(0,line.find(space)+space.length());
		        token = line.substr(0,line.find(space));
		        values[index] = strtod(token.c_str(),NULL);
		        index++;
      		}
      		currentLine++;
   		}
  	}

  	int m=rows[0];
  	for(int i=0;i<N;i++){
    	if(rows[i]!=m){
    	  	m=rows[i];
      		if(m!=columns[i]){
        		cout<<"WARNING: Matrix does not contain a diagonal entry in every row/column"<<endl;
        		return 0;
      		}
    	}
  	}

  	int Nt = 2*N-(Nr-1);  //number of entries in total sparse matrix
  	int *row_total = new int[Nt];
  	int *col_total = new int[Nt];
  	double *val_total = new double[Nt];

  	m = rows[0];
  	index = 0;
  	int i=0;
  	while(i<N){
    	if(rows[i]!=m){
      		for(int j=0;j<i;j++){
        		if(columns[j]==rows[i]){
          			row_total[index] = rows[i];  
          			col_total[index] = rows[j];
          			val_total[index] = values[j];   
          			index++;    
        		}
      		}
      		m=rows[i];
    	}
    	else{
      		row_total[index] = rows[i];
      		col_total[index] = columns[i];
      		val_total[index] = values[i];
      		index++;
      		i++;
    	}
  	}

  	int *row_ptr = new int[Nr];
  	int j=0;
  	int count = 0;
  	row_ptr[0] = 0;
  	row_ptr[Nr-1] = Nt;
  	for(int i=1;i<Nt;i++){
    	if(row_total[i-1]==row_total[i]){
      		count++;
    	}
    	else{
      		count++;
      		j++;
      		row_ptr[j] = count;
    	}    
  	} 



  	//begin test
    srand(0);
    double *x = new double[Nr-1];
    double *b = new double[Nr-1];
    for(int i=0;i<Nr-1;i++){
      x[i] = 2*(rand()%Nr)/(double)Nr-1;
      b[i] = 1.0;
      //if(i<(Nr-1)/2){b[i] = 10.0;}
      //else{b[i] = -5.0;}
    }
    switch(solver)
    {
      case 0:
        std::cout << "Begin test for AMG: " + filename << std::endl;
        amg(row_ptr,col_total,val_total,x,b,Nr-1,0.25,10e-8);
        break;
      case 1:
        //std::cout << "Begin test for PCG: " + filename << std::endl;
        //int niter = pcg(row_ptr,col_total,val_total,x,b,Nr-1,10e-8,100000);
        break;
    }

    //cout<<""<<endl;
    //for(int i=0;i<10;i++){  //Nr-1
    //  cout<<x[i]<<endl;
    //}
  	
  	//amg(row_ptr,col_total,val_total,x,b,Nr-1,0.25,10e-8);
  	//int niter = pcg(row_ptr,col_total,val_total,x,b,Nr-1,10e-8,100000);
  	//int niter = rich(row_ptr,col_total,val_total,x,b,Nr-1,1.0/980,10e-8,100000);
  	//int niter = sor(row_ptr,col_total,val_total,x,b,Nr-1,1.2,10e-8,100000);
  	//int niter = gs(row_ptr,col_total,val_total,x,b,Nr-1,10e-8,100000);
  	//int niter = jac(row_ptr,col_total,val_total,x,b,Nr-1,10e-8,100000);
  	//int niter = sgs(row_ptr,col_total,val_total,x,b,Nr-1,10e-8,100000);
  	//int niter = ssor(row_ptr,col_total,val_total,x,b,Nr-1,1.2,10e-8,100000);
  	//cout<<"niter: "<<niter<<endl;
  	//output b array
  	ofstream myfile("exact.txt");
  	if(myfile.is_open()){
  	 for(int i=0;i<Nr-1;i++){
  	   myfile<<x[i]<<endl;
  	 }
  	}
  	myfile.close();


  	delete[] rows;
  	delete[] columns;
  	delete[] values;
  	delete[] row_total;
  	delete[] col_total;
  	delete[] val_total;
  	delete[] row_ptr;
  	delete[] x;
  	delete[] b;

    return 0;
}