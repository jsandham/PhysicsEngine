#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "MeshLoader.h"

using namespace PhysicsEngine;

bool MeshLoader::load(const std::string& filepath, std::vector<float>& vertices, std::vector<float>& normals, std::vector<float>& texCoords)
{
	int period = filepath.find_last_of(".");
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
				p.push_back((float)std::stod(wordsInLine[i]));
			}
			else if (wordsInLine[0] == "l"){
				l.push_back((float)std::stod(wordsInLine[i]));
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
		f_vn.resize(f_v.size(), 0.0f);

		for (int i = 0; i<f_v.size(); i += 3){
			float f1 = f_v[i];
			float f2 = f_v[i + 1];
			float f3 = f_v[i + 2];

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