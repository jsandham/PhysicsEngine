#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "MeshLoader.h"

using namespace PhysicsEngine;

const int MAX_NUM_GROUPS = 20;
const int MAX_NUM_ELEM_TYP = 20;
const int MAX_CHARS_PER_LINE = 512;
const int MAX_TOKENS_PER_LINE = 20;
const char* DELIMITER = " ";

bool MeshLoader::load(const std::string& filepath, std::vector<float>& vertices, std::vector<float>& normals, std::vector<float>& texCoords)
{
	size_t period = filepath.find_last_of(".");
	std::string extension = filepath.substr(period + 1);

	if (extension != "txt"){
		std::cout << "MeshLoader only excepts .obj files. Call to load failed." << std::endl;
		return false;
	}

	//vertices data
	std::vector<float> v;
	std::vector<float> vt;
	std::vector<float> vn;

	// element data
	std::vector<int> p;
	std::vector<int> l;
	std::vector<int> f_v;
	std::vector<int> f_vt;
	std::vector<int> f_vn;

	std::ifstream myfile;
	std::string line;

	myfile.open(filepath.c_str());
	while (getline(myfile, line)){

		std::stringstream ss;

		ss << line;

		std::string word;
		std::vector<std::string> wordsInLine;
		while (getline(ss, word, ' ')){
			if (word != ""){
				wordsInLine.push_back(word);
			}
		}

		ss.clear();

		for (int i = 1; i<wordsInLine.size(); i++){
			if (wordsInLine[0] == "v"){
				v.push_back((float)std::stod(wordsInLine[i]));
			}
			else if (wordsInLine[0] == "vt"){
				vt.push_back((float)std::stod(wordsInLine[i]));
			}
			else if (wordsInLine[0] == "vn"){
				vn.push_back((float)std::stod(wordsInLine[i]));
			}
			else if (wordsInLine[0] == "p")
			{
				p.push_back(std::stoi(wordsInLine[i]));
			}
			else if (wordsInLine[0] == "l"){
				l.push_back(std::stoi(wordsInLine[i]));
			}
			else if (wordsInLine[0] == "f"){
				ss << wordsInLine[i];
				std::vector<std::string> subWords;
				while (getline(ss, word, '/')){
					subWords.push_back(word);
				}

				ss.clear();

				if (subWords.size() == 0 || subWords.size() > 3){
					std::cout << "Incorrect format given for faces. Call to load failed" << std::endl;
					return false;
				}

				f_v.push_back(std::stoi(subWords[0]));

				if (subWords.size() > 1){
					if (subWords[1] != ""){
						f_vt.push_back(std::stoi(subWords[1]));
					}
				}

				if (subWords.size() > 2){
					if (subWords[2] != ""){
						f_vn.push_back(std::stoi(subWords[2]));
					}
				}
			}
		}
	}

	myfile.close();

	if (v.size() == 0){
		std::cout << "File does not contain any vertices. Call to load failed" << std::endl;
		return false;
	}

	if (f_v.size() == 0){
		std::cout << "File does not contain any faces. Call to load failed" << std::endl;
		return false;
	}

	// calculate normals if not given
	if (vn.size() == 0){
		vn.resize(v.size(), 0.0f);
		f_vn.resize(f_v.size(), 0);

		for (int i = 0; i<f_v.size(); i += 3){
			int f1 = f_v[i];
			int f2 = f_v[i + 1];
			int f3 = f_v[i + 2];

			f_vn[i] = f1;
			f_vn[i + 1] = f2;
			f_vn[i + 2] = f3;

			float a_x = v[3 * (f1 - 1)] - v[3 * (f2 - 1)];
			float a_y = v[3 * (f1 - 1) + 1] - v[3 * (f2 - 1) + 1];
			float a_z = v[3 * (f1 - 1) + 2] - v[3 * (f2 - 1) + 2];

			float b_x = v[3 * (f1 - 1)] - v[3 * (f3 - 1)];
			float b_y = v[3 * (f1 - 1) + 1] - v[3 * (f3 - 1) + 1];
			float b_z = v[3 * (f1 - 1) + 2] - v[3 * (f3 - 1) + 2];

			// compute cross c = a x b
			float c_x = a_y * b_z - a_z * b_y;
			float c_y = a_z * b_x - a_x * b_z;
			float c_z = a_x * b_y - a_y * b_x;

			// normalize c
			float cLength = sqrt(c_x*c_x + c_y*c_y + c_z*c_z);
			c_x = c_x / cLength;
			c_y = c_y / cLength;
			c_z = c_z / cLength;

			vn[3 * (f1 - 1)] += c_x;
			vn[3 * (f1 - 1) + 1] += c_y;
			vn[3 * (f1 - 1) + 2] += c_z;

			vn[3 * (f2 - 1)] += c_x;
			vn[3 * (f2 - 1) + 1] += c_y;
			vn[3 * (f2 - 1) + 2] += c_z;

			vn[3 * (f3 - 1)] += c_x;
			vn[3 * (f3 - 1) + 1] += c_y;
			vn[3 * (f3 - 1) + 2] += c_z;
		}

		for (int i = 0; i<vn.size(); i += 3){
			float v_x = vn[i];
			float v_y = vn[i + 1];
			float v_z = vn[i + 2];

			float vLength = sqrt(v_x*v_x + v_y*v_y + v_z*v_z);
			vn[i] = vn[i] / vLength;
			vn[i + 1] = vn[i + 1] / vLength;
			vn[i + 2] = vn[i + 2] / vLength;
		}
	}

	vertices.clear();
	normals.clear();
	texCoords.clear();

	// set triangle vertices, texture coords, and normals
	for (int i = 0; i<f_v.size(); i++){
		vertices.push_back(v[3 * (f_v[i] - 1)]);
		vertices.push_back(v[3 * (f_v[i] - 1) + 1]);
		vertices.push_back(v[3 * (f_v[i] - 1) + 2]);
	}

	for (int i = 0; i<f_vt.size(); i++){
		texCoords.push_back(vt[2 * (f_vt[i] - 1)]);
		texCoords.push_back(vt[2 * (f_vt[i] - 1) + 1]);
	}

	for (int i = 0; i<f_vn.size(); i++){
		normals.push_back(vn[3 * (f_vn[i] - 1)]);
		normals.push_back(vn[3 * (f_vn[i] - 1) + 1]);
		normals.push_back(vn[3 * (f_vn[i] - 1) + 2]);
	}

	std::cout << vertices.size() << " " << normals.size() << " " << texCoords.size() << std::endl;

	return true;
}


bool MeshLoader::load_gmesh(const std::string& filepath, std::vector<float>& vertices, std::vector<int>& connect, std::vector<int>& bconnect, std::vector<int>& groups)
{
	//create a file-reading object
	std::ifstream myfile;
	myfile.open(filepath.c_str());

	int flag=0,index=0,typ=0,grp=0;
	int tl[MAX_NUM_ELEM_TYP]={}; //list of element types
	int gl[MAX_NUM_GROUPS]={};   //list of groups

	int Dim = 0;     //dimension of problem
	int Ng = 0;      //number of element groups
	int N = 0;       //number of points 
	int Nte = 0;     //total number of elements 
	int Ne = 0;      //number of interior elements
	int Ne_b = 0;    //number of boundary elements
	int Npe = 0;     //number of points per interior element
	int Npe_b = 0;   //number of points per boundary element
	int Type = 0;    //interior element type
	int Type_b = 0;  //boundary element type

	//scan through file (first pass)
	while(!myfile.eof())
	{
		char buf[MAX_CHARS_PER_LINE];
		myfile.getline(buf,MAX_CHARS_PER_LINE);

		const char* token[MAX_TOKENS_PER_LINE]={};

		int n = 0;

		// parse the line
		token[0] = strtok(buf,DELIMITER);
		if(token[0]){
	 		if(strcmp(token[0],"$Nodes")==0) {flag=1; index=-2;}
	  		if(strcmp(token[0],"$Elements")==0) {flag=2; index=-2;}
		  	if(strcmp(token[0],"$EndNodes")==0) {flag=0; index=-2;}
		 	if(strcmp(token[0],"$EndElements")==0) {flag=0; index=-2;}
		  	for(n=1;n<MAX_TOKENS_PER_LINE;n++){
	    		token[n] = strtok(0,DELIMITER);
	   			if(!token[n]) break;
	  		}
		}

		//process the line
		if(index==-1){
	 		if(flag==1) {N = atoi(token[0]);}
	  		if(flag==2) {Nte = atoi(token[0]);}
		}
		else if(index>-1){
	  		if(flag==2){
	    		typ = atoi(token[1]);  //element type
	    		grp = atoi(token[3]);  //group
	    		if(typ==15){           //15 corresponds to a point
	    		}
	    		else if(typ>Type){
	     			Type=typ;
	     			Ne = 1;
	    		}
	    		else if(typ==Type){
	      			Ne++;
	    		}
	    		for(int i=0;i<MAX_NUM_GROUPS;i++){
	      			if(gl[i]==0) {gl[i]=grp; Ng++; break;}
	      			if(gl[i]==grp) {break;}
	    		}
	    		for(int i=0;i<MAX_NUM_ELEM_TYP;i++){
	      			if(tl[i]==0) {tl[i]=typ; break;}
	      			if(tl[i]==typ) {break;}
	    		}
	  		}
		}
		index++;
	}
	Ne_b = Nte-Ne;

	switch (Type)
	{
		case 1:      //linear 1D lines
			Npe = 2;
		    Npe_b = 1;
		    Type_b = 15;
		    Dim = 1;
		    break;
		case 2:      //linear 2D triangles
		    Npe = 3;
		    Npe_b = 2;
		    Type_b = 1;
		    Dim = 2;
		    break;
		case 3:      //linear 2D quadrangles
		    Npe = 4;
		    Npe_b = 2;
		    Type_b = 1;
		    Dim = 2;
		    break;
		case 4:      //linear 3D tetrahedra
		    Npe = 4;
		    Npe_b = 3;
		    Type_b = 2;
		    Dim = 3;
		    break;
		case 8:      //quadratic 1D lines
		    Npe = 3;
		    Npe_b = 1;
		    Type_b =15;
		    Dim = 1;
		    break;
		case 9:      //quadratic 2D triangles
		    Npe = 6;
		    Npe_b = 3;
		    Type_b = 8;
		    Dim = 2;
		    break;
		case 10:     //quadratic 2D quadrangles
		    Npe = 8;
		    Npe_b = 3;
		    Type_b = 8;
		    Dim = 2;
		    break;
		case 11:     //quadratic 3D tetrahedra
		    Npe = 10;
		    Npe_b = 6;
		    Type_b = 9;
		    Dim = 3;
		    break;
	}

	//initialize model
	groups.resize(Ng);
	vertices.resize(3*N);
	connect.resize(Npe*Ne);
	bconnect.resize(Ne_b*(Npe_b+1));

	for(int i=0;i<Ng;i++){groups[i] = gl[i];}

	//return to beginning of file
	myfile.clear();
	myfile.seekg(0,myfile.beg);

	//scan through file (second pass)
	index=0; flag=0;
	while(!myfile.eof())
	{
		char buf[MAX_CHARS_PER_LINE];
		myfile.getline(buf,MAX_CHARS_PER_LINE);

		const char* token[MAX_TOKENS_PER_LINE]={};

		int n = 0;

		// parse the line
		token[0] = strtok(buf,DELIMITER);
		if(token[0]){
	  		if(strcmp(token[0],"$Nodes")==0) {flag=1; index=-2;}
	  		if(strcmp(token[0],"$Elements")==0) {flag=2; index=-2;}
	  		if(strcmp(token[0],"$EndNodes")==0) {flag=0; index=-2;}
	  		if(strcmp(token[0],"$EndElements")==0) {flag=0; index=-2;}
	  		for(n=1;n<MAX_TOKENS_PER_LINE;n++){
	    		token[n] = strtok(0,DELIMITER);
	    		if(!token[n]) break;
	  		}
		}

		//process the line (fill model arrays)
		if(index>-1){
	  		if(flag==1){
	  			vertices[3*index] = strtof(token[1],NULL);
	  			vertices[3*index + 1] = strtof(token[2],NULL);
	  			vertices[3*index + 2] = strtof(token[3],NULL);
	  		}
	  		if(flag==2){
	    		if(atoi(token[1])==Type){
		      		switch (Type)
		      		{
		      			case 1:       //2-point 1D line
		      				connect[2*(index-Ne_b)] = atoi(token[5]);
		        			connect[2*(index-Ne_b) + 1] = atoi(token[6]);
		        			break;
		      			case 2:       //3-point 2D triangle
		      				connect[3*(index-Ne_b)] = atoi(token[5]);
		        			connect[3*(index-Ne_b) + 1] = atoi(token[6]);
		        			connect[3*(index-Ne_b) + 2] = atoi(token[7]);
		        			break;
		      			case 3:       //4-point 2D quadrangle
		      				connect[4*(index-Ne_b)] = atoi(token[5]);
					        connect[4*(index-Ne_b) + 1] = atoi(token[6]);
					        connect[4*(index-Ne_b) + 2] = atoi(token[7]);
					        connect[4*(index-Ne_b) + 3] = atoi(token[8]);
					        break;
		      			case 4:       //4-point 3D tetrahedra
		      				connect[4*(index-Ne_b)] = atoi(token[5]);
					        connect[4*(index-Ne_b) + 1] = atoi(token[6]);
					        connect[4*(index-Ne_b) + 2] = atoi(token[7]);
					        connect[4*(index-Ne_b) + 3] = atoi(token[8]);
					        break;
				      	case 8:       //3-point 1D line
				      		connect[3*(index-Ne_b)] = atoi(token[5]);
					        connect[3*(index-Ne_b) + 1] = atoi(token[6]);
					        connect[3*(index-Ne_b) + 2] = atoi(token[7]);
					        break;
		      			case 9:       //6-point 2D triangle
		      				connect[6*(index-Ne_b)] = atoi(token[5]);
					        connect[6*(index-Ne_b) + 1] = atoi(token[6]);
					        connect[6*(index-Ne_b) + 2] = atoi(token[7]);
					        connect[6*(index-Ne_b) + 3] = atoi(token[8]);
					        connect[6*(index-Ne_b) + 4] = atoi(token[9]);
					        connect[6*(index-Ne_b) + 5] = atoi(token[10]);
					        break;
		      			case 10:      //8-point 2D quadrangle
		      				connect[8*(index-Ne_b)] = atoi(token[5]);
					        connect[8*(index-Ne_b) + 1] = atoi(token[6]);
					        connect[8*(index-Ne_b) + 2] = atoi(token[7]);
					        connect[8*(index-Ne_b) + 3] = atoi(token[8]);
					        connect[8*(index-Ne_b) + 4] = atoi(token[9]);
					        connect[8*(index-Ne_b) + 5] = atoi(token[10]);
					        connect[8*(index-Ne_b) + 6] = atoi(token[11]);
					        connect[8*(index-Ne_b) + 7] = atoi(token[12]);
					        break;
		      			case 11:      //10-point 3D tetrahedra
		      				connect[10*(index-Ne_b)] = atoi(token[5]);
					        connect[10*(index-Ne_b) + 1] = atoi(token[6]);
					        connect[10*(index-Ne_b) + 2] = atoi(token[7]);
					        connect[10*(index-Ne_b) + 3] = atoi(token[8]);
					        connect[10*(index-Ne_b) + 4] = atoi(token[9]);
					        connect[10*(index-Ne_b) + 5] = atoi(token[10]);
					        connect[10*(index-Ne_b) + 6] = atoi(token[11]);
					        connect[10*(index-Ne_b) + 7] = atoi(token[12]);
					        connect[10*(index-Ne_b) + 8] = atoi(token[13]);
					        connect[10*(index-Ne_b) + 9] = atoi(token[14]);
					        break;
		      		}
		    	}
	    		else{
		      		switch (Type)
		      		{
		      			case 1:       //1-point (2-point 1D line)
		        			bconnect[2*index] = atoi(token[3]);
					        bconnect[2*index + 1] = atoi(token[5]);
					        break;
		      			case 2:       //2-point line (3-point 2D triangle)
					        bconnect[3*index] = atoi(token[3]);
					        bconnect[3*index + 1] = atoi(token[5]);
					        bconnect[3*index + 2] = atoi(token[6]);
					        break;
		      			case 3:       //2-point line (4-point 2D quadrangle)
		      				bconnect[3*index] = atoi(token[3]);
					        bconnect[3*index + 1] = atoi(token[5]);
					        bconnect[3*index + 2] = atoi(token[6]);
					        break;
		      			case 4:       //3-point triangle (4-point 3D tetrahedra)
		      				bconnect[4*index] = atoi(token[3]);
					        bconnect[4*index + 1] = atoi(token[5]);
					        bconnect[4*index + 2] = atoi(token[6]);
					        bconnect[4*index + 3] = atoi(token[7]);
					        break;
		      			case 8:       //1-point (3-point 1D line)
		      				bconnect[2*index] = atoi(token[3]);
					        bconnect[2*index + 1] = atoi(token[5]);
					        break;
		      			case 9:       //3-point line (6-point 2D triangle)
		      				bconnect[4*index] = atoi(token[3]);
					        bconnect[4*index + 1] = atoi(token[5]);
					        bconnect[4*index + 2] = atoi(token[6]);
					        bconnect[4*index + 3] = atoi(token[7]);
					        break;
		      			case 10:      //3-point line (8-point 2D quadrangle)
		      				bconnect[4*index] = atoi(token[3]);
					        bconnect[4*index + 1] = atoi(token[5]);
					        bconnect[4*index + 2] = atoi(token[6]);
					        bconnect[4*index + 3] = atoi(token[7]);
					        break;
		      			case 11:      //6-point triangle (10-point 3D tetrahedra)
		      				bconnect[7*index] = atoi(token[3]);
					        bconnect[7*index + 1] = atoi(token[5]);
					        bconnect[7*index + 2] = atoi(token[6]);
					        bconnect[7*index + 3] = atoi(token[7]);
					        bconnect[7*index + 4] = atoi(token[8]);
					        bconnect[7*index + 5] = atoi(token[9]);
					        bconnect[7*index + 6] = atoi(token[10]);
					        break;
	      			}
	    		}
	  		}
		}
		index++;
	}
	myfile.close();

	return true;
}