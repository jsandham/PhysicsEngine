#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "../../include/stb_image/stb_image.h"

#include "../../include/core/AssetLoader.h"

using namespace PhysicsEngine;

bool AssetLoader::load(const std::string& filepath, Shader& shader)
{
	std::ifstream in(filepath.c_str());
	std::ostringstream contents; contents << in.rdbuf(); in.close();

	std::string shaderContent = contents.str();

	std::string vertexTag = "VERTEX:";
	std::string geometryTag = "GEOMETRY:";
	std::string fragmentTag = "FRAGMENT:";

	size_t startOfVertexTag = shaderContent.find(vertexTag, 0);
	size_t startOfGeometryTag = shaderContent.find(geometryTag, 0);
	size_t startOfFragmentTag = shaderContent.find(fragmentTag, 0);

	if(startOfVertexTag == std::string::npos || startOfFragmentTag == std::string::npos){
		std::cout << "Error: Shader must contain both a vertex shader and a fragment shader" << std::endl;
		return false;
	}

	std::string vertexShader, geometryShader, fragmentShader;

	if(startOfGeometryTag == std::string::npos){
		vertexShader = shaderContent.substr(startOfVertexTag + vertexTag.length(), startOfFragmentTag - vertexTag.length());
		geometryShader = "";
		fragmentShader = shaderContent.substr(startOfFragmentTag + fragmentTag.length(), shaderContent.length());
	}
	else{
		vertexShader = shaderContent.substr(startOfVertexTag + vertexTag.length(), startOfGeometryTag - vertexTag.length());
		geometryShader = shaderContent.substr(startOfGeometryTag + geometryTag.length(), startOfFragmentTag - geometryTag.length());
		fragmentShader = shaderContent.substr(startOfFragmentTag + fragmentTag.length(), shaderContent.length());
	}

	// trim left
	size_t firstNotOfIndex;
	firstNotOfIndex = vertexShader.find_first_not_of("\n");
	if(firstNotOfIndex != std::string::npos){
		vertexShader = vertexShader.substr(firstNotOfIndex);
	}

	firstNotOfIndex = geometryShader.find_first_not_of("\n");
	if(firstNotOfIndex != std::string::npos){
		geometryShader = geometryShader.substr(firstNotOfIndex);
	}

	firstNotOfIndex = fragmentShader.find_first_not_of("\n");
	if(firstNotOfIndex != std::string::npos){
		fragmentShader = fragmentShader.substr(firstNotOfIndex);
	}

	// trim right
	size_t lastNotOfIndex;
	lastNotOfIndex = vertexShader.find_last_not_of("\n");
	if(lastNotOfIndex != std::string::npos){
		vertexShader.erase(lastNotOfIndex + 1);
	}

	lastNotOfIndex = geometryShader.find_last_not_of("\n");
	if(lastNotOfIndex != std::string::npos){
		geometryShader.erase(lastNotOfIndex + 1);
	}

	lastNotOfIndex = fragmentShader.find_last_not_of("\n");
	if(lastNotOfIndex != std::string::npos){
		fragmentShader.erase(lastNotOfIndex + 1);
	}

	shader.vertexShader = vertexShader;
	shader.geometryShader = geometryShader;
	shader.fragmentShader = fragmentShader;

	return true;
}

bool AssetLoader::load(const std::string& filepath, Texture2D& texture)
{
	int width, height, numChannels;
	unsigned char* raw = stbi_load(filepath.c_str(), &width, &height, &numChannels, 0);

	if(raw != NULL){
		int size = width * height * numChannels;

		std::vector<unsigned char> data;
		data.resize(size);

		for(unsigned int j = 0; j < data.size(); j++){ data[j] = raw[j]; }

		stbi_image_free(raw);

		TextureFormat format;
		switch(numChannels)
		{
			case 1:
				format = TextureFormat::Depth;
				break;
			case 2:
				format = TextureFormat::RG;
				break;
			case 3:
				format = TextureFormat::RGB;
				break;
			case 4:
				format = TextureFormat::RGBA;
				break;
			default:
				std::cout << "Error: Unsupported number of channels (" << numChannels << ") found when loading texture " << filepath << std::endl;
				return false;
		}

		texture.setRawTextureData(data, width, height, format);
	}
	else{
		std::cout << "Error: stbi_load failed to load texture " << filepath << " with reported reason: " << stbi_failure_reason() << std::endl;
		return false;
	}

	return true;
}


void AssetLoader::split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (getline(ss, item, delim)) {
		if(item.length() > 0){
			elems.push_back(item);
		}
	}
}

std::vector<std::string> AssetLoader::split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	elems.reserve(10);
	split(s, delim, elems);
	return elems;
}

bool AssetLoader::load(const std::string& filepath, Mesh& mesh)
{
	std::ifstream file;
	file.open(filepath, std::ios::in);

	if (!file.is_open()){
		std::cout << "Error: Could not open file " << filepath << std::endl;
		return false;
	}

	std::vector<float> v;
	std::vector<float> vn;
	std::vector<float> vt;
	std::vector<int> f_v;
	std::vector<int> f_vt;
	std::vector<int> f_vn;
	std::vector<std::string> g;
	std::vector<std::string> mtllib;
	std::vector<std::string> usemtl;

	int faceCount = 0;
	std::vector<int> subMeshFaceStartIndices;

	std::vector<int> faceStartIndices;
	faceStartIndices.push_back(0);

	bool error = false;
	std::string errorString = "";

	int lineNumber = 0;
	std::string line;
	while (!file.eof()){
		lineNumber++;
		getline(file, line);

		//std::cout << "line: '" << line << "'" << std::endl;

		std::vector< std::string > words = split(line, ' ');

		if (words.size() == 0){ continue; }

		size_t wordsSize = words.size();

		std::string command = words[0];
		//std::cout << "Parsing command: " << command << std::endl;

		if (command == "v"){
			if (!(wordsSize - 1 == 3 || wordsSize - 1 == 4)){
				error = true;
				errorString = "vertex at line number " + std::to_string(lineNumber) + " must contain x y z (w) components but has " + std::to_string(wordsSize);
			}

			for (size_t i = 1; i < words.size(); i++){
				float number;
				std::stringstream(words[i]) >> number;
				v.push_back(number);
			}

			if (wordsSize - 1 == 3){
				v.push_back(1.0);
			}
		}
		else if (command == "vn"){
			if (!(wordsSize - 1 == 3)){
				error = true;
				errorString = "normal at line number " + std::to_string(lineNumber) + " must contain exactly i j k components";
			}

			for (size_t i = 1; i < words.size(); i++){
				float number;
				std::stringstream(words[i]) >> number;
				vn.push_back(number);
			}
		}
		else if (command == "vt"){
			if (!(wordsSize - 1 == 1 || wordsSize - 1 == 2 || wordsSize - 1 == 3)){
				error = true;
				errorString = "texture coordinate at line number " + std::to_string(lineNumber) + " must contain u (v) (w) components";
			}

			for (size_t i = 1; i < words.size(); i++){
				float number;
				std::stringstream(words[i]) >> number;
				vt.push_back(number);
			}

			for (size_t i = words.size(); i < 4; i++){
				vt.push_back(0.0f);
			}
		}
		else if (command == "f"){
			faceCount++;
			faceStartIndices.push_back(faceStartIndices.back() + (int)words.size() - 1);

			for (size_t i = 1; i < words.size(); i++){
				std::string word = words[i];

				int firstSlashIndex = -1;
				int secondSlashIndex = -1;
				size_t index = 0;
				while (index < word.size()){
					if (word[index] == '/'){
						firstSlashIndex = (int)index;
						break;
					}

					index++;
				}

				index++;
				while (index < word.size()){
					if (word[index] == '/'){
						secondSlashIndex = (int)index;
						break;
					}

					index++;
				}

				if (firstSlashIndex == -1 && secondSlashIndex != -1){
					error = true;
					errorString = "Error: Could not find first slash index but could find second when parsing faces";
				}

				std::string vStr = word;
				std::string vtStr = "";
				std::string vnStr = "";

				//std::cout << "first slash index: " << firstSlashIndex << " second slash index: " << secondSlashIndex << std::endl;
			
				if (firstSlashIndex != -1 && secondSlashIndex != -1 /*2 slashes exist*/){
					vStr = word.substr(0, firstSlashIndex - 0);
					if (secondSlashIndex - firstSlashIndex > 1){
						vtStr = word.substr(firstSlashIndex + 1, secondSlashIndex - firstSlashIndex - 1);
					}
					vnStr = word.substr(secondSlashIndex + 1, word.size() - secondSlashIndex - 1);
				}
				else if (firstSlashIndex != -1 && secondSlashIndex == -1 /*1 slash exists*/){
					vStr = word.substr(0, firstSlashIndex - 0);
					vtStr = word.substr(firstSlashIndex + 1, secondSlashIndex - firstSlashIndex - 1);
				}

				//TODO: Check that vStr, vtStr, and vnStr are valid numbers
				int number;
				std::istringstream(vStr) >> number;
				f_v.push_back(number);
				if (vtStr.length() > 0){
					std::istringstream(vtStr) >> number;
					f_vt.push_back(number);
				}
				if (vnStr.length() > 0){
					std::istringstream(vnStr) >> number;
					f_vn.push_back(number);
				}
			}

			if (f_vt.size() != f_v.size() && f_vt.size() > 0){
				error = true;
				errorString = "Error: Incorrect number of texture coordinates found in faces";
			}
			if (f_vn.size() != f_v.size() && f_vn.size() > 0){
				error = true;
				errorString = "Error: Incorrect number of normals found in faces";
			}
		}
		else if (command == "g"){
		}
		else if (command == "usemtl"){
			subMeshFaceStartIndices.push_back(faceCount);
		}

		if (error){
			break;
		}
	}

	subMeshFaceStartIndices.push_back(faceCount);

	// calculate normals if not given
	if (vn.size() == 0){
		vn.resize(3*(v.size() / 4), 0.0f); // v can have 4 components per vertex by the obj standard
		f_vn.resize(f_v.size(), 0);

		for (int i = 0; i < f_v.size(); i += 3){
			int f1 = f_v[i];
			int f2 = f_v[i + 1];
			int f3 = f_v[i + 2];

			f_vn[i] = f1;
			f_vn[i + 1] = f2;
			f_vn[i + 2] = f3;

			float a_x = v[4 * (f1 - 1)] - v[4 * (f2 - 1)];
			float a_y = v[4 * (f1 - 1) + 1] - v[4 * (f2 - 1) + 1];
			float a_z = v[4 * (f1 - 1) + 2] - v[4 * (f2 - 1) + 2];

			float b_x = v[4 * (f1 - 1)] - v[4 * (f3 - 1)];
			float b_y = v[4 * (f1 - 1) + 1] - v[4 * (f3 - 1) + 1];
			float b_z = v[4 * (f1 - 1) + 2] - v[4 * (f3 - 1) + 2];

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

		for (int i = 0; i < vn.size(); i += 3){
			float v_x = vn[i];
			float v_y = vn[i + 1];
			float v_z = vn[i + 2];

			float vLength = sqrt(v_x*v_x + v_y*v_y + v_z*v_z);
			vn[i] = vn[i] / vLength;
			vn[i + 1] = vn[i + 1] / vLength;
			vn[i + 2] = vn[i + 2] / vLength;
		}
	}

	std::vector<float> vertices;
	std::vector<float> normals;
	std::vector<float> texCoords;
	std::vector<int> subMeshVertexStartIndices;

	// loop through each sub mesh
	for (size_t i = 0; i < subMeshFaceStartIndices.size() - 1; i++){
		int subMeshFaceStartIndex = subMeshFaceStartIndices[i];
		int subMeshFaceEndIndex = subMeshFaceStartIndices[i + 1];

		subMeshVertexStartIndices.push_back((int)vertices.size());

		// loop through all faces in sub mesh
		for (size_t j = subMeshFaceStartIndex; j < subMeshFaceEndIndex; j++){
			int startIndex = faceStartIndices[j];
			int endIndex = faceStartIndices[j + 1];


			// n = 0;
			// triangles[n++] = [values[0], values[1], values[2]];
			// for(i = 3; i < count(values); ++i)
			//   triangles[n++] = [
			//     values[i - 3],
			//     values[i - 1],
			//     values[i]
			//   ];

			if (endIndex - startIndex == 3 /*triangle face*/){
				int f_v1 = f_v[startIndex + 0] - 1;
				int f_v2 = f_v[startIndex + 1] - 1;
				int f_v3 = f_v[startIndex + 2] - 1;

				vertices.push_back(v[4 * f_v1 + 0]);
				vertices.push_back(v[4 * f_v1 + 1]);
				vertices.push_back(v[4 * f_v1 + 2]);

				vertices.push_back(v[4 * f_v2 + 0]);
				vertices.push_back(v[4 * f_v2 + 1]);
				vertices.push_back(v[4 * f_v2 + 2]);

				vertices.push_back(v[4 * f_v3 + 0]);
				vertices.push_back(v[4 * f_v3 + 1]);
				vertices.push_back(v[4 * f_v3 + 2]);

				if (f_vt.size() > 0){
					int f_vt1 = f_vt[startIndex + 0] - 1;
					int f_vt2 = f_vt[startIndex + 1] - 1;
					int f_vt3 = f_vt[startIndex + 2] - 1;

					texCoords.push_back(vt[3 * f_vt1 + 0]);
					texCoords.push_back(vt[3 * f_vt1 + 1]);

					texCoords.push_back(vt[3 * f_vt2 + 0]);
					texCoords.push_back(vt[3 * f_vt2 + 1]);

					texCoords.push_back(vt[3 * f_vt3 + 0]);
					texCoords.push_back(vt[3 * f_vt3 + 1]);
				}

				if (f_vn.size() > 0){
					int f_vn1 = f_vn[startIndex + 0] - 1;
					int f_vn2 = f_vn[startIndex + 1] - 1;
					int f_vn3 = f_vn[startIndex + 2] - 1;

					normals.push_back(vn[3 * f_vn1 + 0]);
					normals.push_back(vn[3 * f_vn1 + 1]);
					normals.push_back(vn[3 * f_vn1 + 2]);

					normals.push_back(vn[3 * f_vn2 + 0]);
					normals.push_back(vn[3 * f_vn2 + 1]);
					normals.push_back(vn[3 * f_vn2 + 2]);

					normals.push_back(vn[3 * f_vn3 + 0]);
					normals.push_back(vn[3 * f_vn3 + 1]);
					normals.push_back(vn[3 * f_vn3 + 2]);
				}
			}
			else if (endIndex - startIndex == 4 /*quadrilateral face*/){
				int f_v1 = f_v[startIndex + 0] - 1;
				int f_v2 = f_v[startIndex + 1] - 1;
				int f_v3 = f_v[startIndex + 2] - 1;
				int f_v4 = f_v[startIndex + 3] - 1;

				vertices.push_back(v[4 * f_v1 + 0]);
				vertices.push_back(v[4 * f_v1 + 1]);
				vertices.push_back(v[4 * f_v1 + 2]);

				vertices.push_back(v[4 * f_v2 + 0]);
				vertices.push_back(v[4 * f_v2 + 1]);
				vertices.push_back(v[4 * f_v2 + 2]);

				vertices.push_back(v[4 * f_v3 + 0]);
				vertices.push_back(v[4 * f_v3 + 1]);
				vertices.push_back(v[4 * f_v3 + 2]);

				vertices.push_back(v[4 * f_v1 + 0]);
				vertices.push_back(v[4 * f_v1 + 1]);
				vertices.push_back(v[4 * f_v1 + 2]);

				vertices.push_back(v[4 * f_v3 + 0]);
				vertices.push_back(v[4 * f_v3 + 1]);
				vertices.push_back(v[4 * f_v3 + 2]);

				vertices.push_back(v[4 * f_v4 + 0]);
				vertices.push_back(v[4 * f_v4 + 1]);
				vertices.push_back(v[4 * f_v4 + 2]);

				if (f_vt.size() > 0){
					int f_vt1 = f_vt[startIndex + 0] - 1;
					int f_vt2 = f_vt[startIndex + 1] - 1;
					int f_vt3 = f_vt[startIndex + 2] - 1;
					int f_vt4 = f_vt[startIndex + 3] - 1;

					texCoords.push_back(vt[3 * f_vt1 + 0]);
					texCoords.push_back(vt[3 * f_vt1 + 1]);

					texCoords.push_back(vt[3 * f_vt2 + 0]);
					texCoords.push_back(vt[3 * f_vt2 + 1]);

					texCoords.push_back(vt[3 * f_vt3 + 0]);
					texCoords.push_back(vt[3 * f_vt3 + 1]);

					texCoords.push_back(vt[3 * f_vt1 + 0]);
					texCoords.push_back(vt[3 * f_vt1 + 1]);

					texCoords.push_back(vt[3 * f_vt3 + 0]);
					texCoords.push_back(vt[3 * f_vt3 + 1]);

					texCoords.push_back(vt[3 * f_vt4 + 0]);
					texCoords.push_back(vt[3 * f_vt4 + 1]);
				}

				if (f_vn.size() > 0){
					int f_vn1 = f_vn[startIndex + 0] - 1;
					int f_vn2 = f_vn[startIndex + 1] - 1;
					int f_vn3 = f_vn[startIndex + 2] - 1;
					int f_vn4 = f_vn[startIndex + 3] - 1;

					normals.push_back(vn[3 * f_vn1 + 0]);
					normals.push_back(vn[3 * f_vn1 + 1]);
					normals.push_back(vn[3 * f_vn1 + 2]);

					normals.push_back(vn[3 * f_vn2 + 0]);
					normals.push_back(vn[3 * f_vn2 + 1]);
					normals.push_back(vn[3 * f_vn2 + 2]);

					normals.push_back(vn[3 * f_vn3 + 0]);
					normals.push_back(vn[3 * f_vn3 + 1]);
					normals.push_back(vn[3 * f_vn3 + 2]);

					normals.push_back(vn[3 * f_vn1 + 0]);
					normals.push_back(vn[3 * f_vn1 + 1]);
					normals.push_back(vn[3 * f_vn1 + 2]);

					normals.push_back(vn[3 * f_vn3 + 0]);
					normals.push_back(vn[3 * f_vn3 + 1]);
					normals.push_back(vn[3 * f_vn3 + 2]);

					normals.push_back(vn[3 * f_vn4 + 0]);
					normals.push_back(vn[3 * f_vn4 + 1]);
					normals.push_back(vn[3 * f_vn4 + 2]);
				}
			}
			else{
				error = true;
				errorString = "Error: face (" + std::to_string(faceCount) + ") with " + std::to_string(endIndex - startIndex) + " vertices not currently supported";
			}
		}
	}

	subMeshVertexStartIndices.push_back((int)vertices.size());

	mesh.vertices = vertices;
	mesh.normals = normals;
	mesh.texCoords = texCoords;
	mesh.subMeshVertexStartIndices = subMeshVertexStartIndices;

	if (error){
		std::cout << "Error: " << errorString << std::endl;
		return false;
	}

	std::cout << "done" << std::endl;

	file.close();

	return true;
}









const int MAX_NUM_GROUPS = 20;
const int MAX_NUM_ELEM_TYP = 20;
const int MAX_CHARS_PER_LINE = 512;
const int MAX_TOKENS_PER_LINE = 20;
const char* DELIMITER = " ";

// bool AssetLoader::load(const std::string& filepath, Mesh& mesh)
// {
// 	size_t period = filepath.find_last_of(".");
// 	std::string extension = filepath.substr(period + 1);

// 	if (extension != "txt"){
// 		std::cout << "MeshLoader only excepts .obj files. Call to load failed." << std::endl;
// 		return false;
// 	}

// 	//vertices data
// 	std::vector<float> v;
// 	std::vector<float> vt;
// 	std::vector<float> vn;

// 	// element data
// 	std::vector<int> p;
// 	std::vector<int> l;
// 	std::vector<int> f_v;
// 	std::vector<int> f_vt;
// 	std::vector<int> f_vn;

// 	std::ifstream myfile;
// 	std::string line;

// 	myfile.open(filepath.c_str());
// 	while (getline(myfile, line)){

// 		std::stringstream ss;

// 		ss << line;

// 		std::string word;
// 		std::vector<std::string> wordsInLine;
// 		while (getline(ss, word, ' ')){
// 			if (word != ""){
// 				wordsInLine.push_back(word);
// 			}
// 		}

// 		ss.clear();

// 		for (int i = 1; i<wordsInLine.size(); i++){
// 			if (wordsInLine[0] == "v"){
// 				v.push_back((float)std::stod(wordsInLine[i]));
// 			}
// 			else if (wordsInLine[0] == "vt"){
// 				vt.push_back((float)std::stod(wordsInLine[i]));
// 			}
// 			else if (wordsInLine[0] == "vn"){
// 				vn.push_back((float)std::stod(wordsInLine[i]));
// 			}
// 			else if (wordsInLine[0] == "p")
// 			{
// 				p.push_back(std::stoi(wordsInLine[i]));
// 			}
// 			else if (wordsInLine[0] == "l"){
// 				l.push_back(std::stoi(wordsInLine[i]));
// 			}
// 			else if (wordsInLine[0] == "f"){
// 				ss << wordsInLine[i];
// 				std::vector<std::string> subWords;
// 				while (getline(ss, word, '/')){
// 					subWords.push_back(word);
// 				}

// 				ss.clear();

// 				if (subWords.size() == 0 || subWords.size() > 3){
// 					std::cout << "Incorrect format given for faces. Call to load failed" << std::endl;
// 					return false;
// 				}

// 				f_v.push_back(std::stoi(subWords[0]));

// 				if (subWords.size() > 1){
// 					if (subWords[1] != ""){
// 						f_vt.push_back(std::stoi(subWords[1]));
// 					}
// 				}

// 				if (subWords.size() > 2){
// 					if (subWords[2] != ""){
// 						f_vn.push_back(std::stoi(subWords[2]));
// 					}
// 				}
// 			}
// 		}
// 	}

// 	myfile.close();

// 	if (v.size() == 0){
// 		std::cout << "File does not contain any vertices. Call to load failed" << std::endl;
// 		return false;
// 	}

// 	if (f_v.size() == 0){
// 		std::cout << "File does not contain any faces. Call to load failed" << std::endl;
// 		return false;
// 	}

// 	// calculate normals if not given
// 	if (vn.size() == 0){
// 		vn.resize(v.size(), 0.0f);
// 		f_vn.resize(f_v.size(), 0);

// 		for (int i = 0; i<f_v.size(); i += 3){
// 			int f1 = f_v[i];
// 			int f2 = f_v[i + 1];
// 			int f3 = f_v[i + 2];

// 			f_vn[i] = f1;
// 			f_vn[i + 1] = f2;
// 			f_vn[i + 2] = f3;

// 			float a_x = v[3 * (f1 - 1)] - v[3 * (f2 - 1)];
// 			float a_y = v[3 * (f1 - 1) + 1] - v[3 * (f2 - 1) + 1];
// 			float a_z = v[3 * (f1 - 1) + 2] - v[3 * (f2 - 1) + 2];

// 			float b_x = v[3 * (f1 - 1)] - v[3 * (f3 - 1)];
// 			float b_y = v[3 * (f1 - 1) + 1] - v[3 * (f3 - 1) + 1];
// 			float b_z = v[3 * (f1 - 1) + 2] - v[3 * (f3 - 1) + 2];

// 			// compute cross c = a x b
// 			float c_x = a_y * b_z - a_z * b_y;
// 			float c_y = a_z * b_x - a_x * b_z;
// 			float c_z = a_x * b_y - a_y * b_x;

// 			// normalize c
// 			float cLength = sqrt(c_x*c_x + c_y*c_y + c_z*c_z);
// 			c_x = c_x / cLength;
// 			c_y = c_y / cLength;
// 			c_z = c_z / cLength;

// 			vn[3 * (f1 - 1)] += c_x;
// 			vn[3 * (f1 - 1) + 1] += c_y;
// 			vn[3 * (f1 - 1) + 2] += c_z;

// 			vn[3 * (f2 - 1)] += c_x;
// 			vn[3 * (f2 - 1) + 1] += c_y;
// 			vn[3 * (f2 - 1) + 2] += c_z;

// 			vn[3 * (f3 - 1)] += c_x;
// 			vn[3 * (f3 - 1) + 1] += c_y;
// 			vn[3 * (f3 - 1) + 2] += c_z;
// 		}

// 		for (int i = 0; i<vn.size(); i += 3){
// 			float v_x = vn[i];
// 			float v_y = vn[i + 1];
// 			float v_z = vn[i + 2];

// 			float vLength = sqrt(v_x*v_x + v_y*v_y + v_z*v_z);
// 			vn[i] = vn[i] / vLength;
// 			vn[i + 1] = vn[i + 1] / vLength;
// 			vn[i + 2] = vn[i + 2] / vLength;
// 		}
// 	}

// 	mesh.vertices.clear();
// 	mesh.normals.clear();
// 	mesh.texCoords.clear();
// 	mesh.subMeshStartIndicies.clear();

// 	mesh.subMeshStartIndicies.push_back(0);

// 	// set triangle vertices, texture coords, and normals
// 	for (int i = 0; i<f_v.size(); i++){
// 		mesh.vertices.push_back(v[3 * (f_v[i] - 1)]);
// 		mesh.vertices.push_back(v[3 * (f_v[i] - 1) + 1]);
// 		mesh.vertices.push_back(v[3 * (f_v[i] - 1) + 2]);
// 	}

// 	for (int i = 0; i<f_vt.size(); i++){
// 		mesh.texCoords.push_back(vt[2 * (f_vt[i] - 1)]);
// 		mesh.texCoords.push_back(vt[2 * (f_vt[i] - 1) + 1]);
// 	}

// 	for (int i = 0; i<f_vn.size(); i++){
// 		mesh.normals.push_back(vn[3 * (f_vn[i] - 1)]);
// 		mesh.normals.push_back(vn[3 * (f_vn[i] - 1) + 1]);
// 		mesh.normals.push_back(vn[3 * (f_vn[i] - 1) + 2]);
// 	}

// 	std::cout << mesh.vertices.size() << " " << mesh.normals.size() << " " << mesh.texCoords.size() << std::endl;

// 	return true;
// }

bool AssetLoader::load(const std::string& filepath, GMesh& gmesh)
{
	//create a file-reading object
	std::ifstream myfile;
	myfile.open(filepath.c_str());

	int flag=0,index=0,typ=0,grp=0;
	int tl[MAX_NUM_ELEM_TYP]={}; //list of element types
	int gl[MAX_NUM_GROUPS]={};   //list of groups

	gmesh.dim = 0;             
    gmesh.ng = 0;                 
    gmesh.n = 0;                                   
    gmesh.nte = 0;                  
    gmesh.ne = 0;                              
    gmesh.ne_b = 0;                                             
    gmesh.npe = 0;                     
    gmesh.npe_b = 0;                
    gmesh.type = 0;                              
    gmesh.type_b = 0;             

	//scan through file (first pass)
	while(!myfile.eof())
	{
		char buf[MAX_CHARS_PER_LINE];
		myfile.getline(buf,MAX_CHARS_PER_LINE);

		const char* token[MAX_TOKENS_PER_LINE]={};

		// parse the line
		token[0] = strtok(buf,DELIMITER);
		if(token[0]){
	 		if(strcmp(token[0],"$Nodes")==0) {flag=1; index=-2;}
	  		if(strcmp(token[0],"$Elements")==0) {flag=2; index=-2;}
		  	if(strcmp(token[0],"$EndNodes")==0) {flag=0; index=-2;}
		 	if(strcmp(token[0],"$EndElements")==0) {flag=0; index=-2;}
		  	
		  	for(int k = 1; k < MAX_TOKENS_PER_LINE; k++){
	    		token[k] = strtok(0, DELIMITER);
	   			if(!token[k]) break;
	  		}
		}

		//process the line
		if(index==-1){
	 		if(flag==1) {gmesh.n = atoi(token[0]);}
	  		if(flag==2) {gmesh.nte = atoi(token[0]);}
		}
		else if(index>-1){
	  		if(flag==2){
	    		typ = atoi(token[1]);  //element type
	    		grp = atoi(token[3]);  //group
	    		if(typ == 15){           //15 corresponds to a point
	    		}
	    		else if(typ > gmesh.type){
	     			gmesh.type = typ;
	     			gmesh.ne = 1;
	    		}
	    		else if(typ == gmesh.type){
	      			gmesh.ne++;
	    		}
	    		for(int i = 0; i < MAX_NUM_GROUPS; i++){
	      			if(gl[i] == 0) {gl[i] = grp; gmesh.ng++; break;}
	      			if(gl[i] == grp) {break;}
	    		}
	    		for(int i = 0; i < MAX_NUM_ELEM_TYP; i++){
	      			if(tl[i] == 0) {tl[i] = typ; break;}
	      			if(tl[i] == typ) {break;}
	    		}
	  		}
		}
		index++;
	}
	gmesh.ne_b = gmesh.nte - gmesh.ne;

	switch (gmesh.type)
	{
		case 1:      //linear 1D lines
			gmesh.npe = 2;
		    gmesh.npe_b = 1;
		    gmesh.type_b = 15;
		    gmesh.dim = 1;
		    break;
		case 2:      //linear 2D triangles
		    gmesh.npe = 3;
		    gmesh.npe_b = 2;
		    gmesh.type_b = 1;
		    gmesh.dim = 2;
		    break;
		case 3:      //linear 2D quadrangles
		    gmesh.npe = 4;
		    gmesh.npe_b = 2;
		    gmesh.type_b = 1;
		    gmesh.dim = 2;
		    break;
		case 4:      //linear 3D tetrahedra
		    gmesh.npe = 4;
		    gmesh.npe_b = 3;
		    gmesh.type_b = 2;
		    gmesh.dim = 3;
		    break;
		case 8:      //quadratic 1D lines
		    gmesh.npe = 3;
		    gmesh.npe_b = 1;
		    gmesh.type_b =15;
		    gmesh.dim = 1;
		    break;
		case 9:      //quadratic 2D triangles
		    gmesh.npe = 6;
		    gmesh.npe_b = 3;
		    gmesh.type_b = 8;
		    gmesh.dim = 2;
		    break;
		case 10:     //quadratic 2D quadrangles
		    gmesh.npe = 8;
		    gmesh.npe_b = 3;
		    gmesh.type_b = 8;
		    gmesh.dim = 2;
		    break;
		case 11:     //quadratic 3D tetrahedra
		    gmesh.npe = 10;
		    gmesh.npe_b = 6;
		    gmesh.type_b = 9;
		    gmesh.dim = 3;
		    break;
	}

	//initialize model
	gmesh.groups.resize(gmesh.ng);
	gmesh.vertices.resize(3*gmesh.n);
	gmesh.connect.resize(gmesh.npe*gmesh.ne);
	gmesh.bconnect.resize(gmesh.ne_b*(gmesh.npe_b+1));

	for(int i = 0; i < gmesh.ng; i++){gmesh.groups[i] = gl[i];}

	//return to beginning of file
	myfile.clear();
	myfile.seekg(0, myfile.beg);

	//scan through file (second pass)
	index = 0; flag = 0;
	while(!myfile.eof())
	{
		char buf[MAX_CHARS_PER_LINE];
		myfile.getline(buf,MAX_CHARS_PER_LINE);

		const char* token[MAX_TOKENS_PER_LINE]={};

		// parse the line
		token[0] = strtok(buf,DELIMITER);
		if(token[0]){
	  		if(strcmp(token[0],"$Nodes")==0) {flag=1; index=-2;}
	  		if(strcmp(token[0],"$Elements")==0) {flag=2; index=-2;}
	  		if(strcmp(token[0],"$EndNodes")==0) {flag=0; index=-2;}
	  		if(strcmp(token[0],"$EndElements")==0) {flag=0; index=-2;}
	  		for(int k = 1; k < MAX_TOKENS_PER_LINE; k++){
	    		token[k] = strtok(0, DELIMITER);
	    		if(!token[k]) break;
	  		}
		}

		//process the line (fill model arrays)
		if(index > -1){
	  		if(flag == 1){
	  			gmesh.vertices[3*index] = strtof(token[1],NULL);
	  			gmesh.vertices[3*index + 1] = strtof(token[2],NULL);
	  			gmesh.vertices[3*index + 2] = strtof(token[3],NULL);
	  		}
	  		if(flag == 2){
	    		if(atoi(token[1]) == gmesh.type){
		      		switch (gmesh.type)
		      		{
		      			case 1:       //2-point 1D line
		      				gmesh.connect[2*(index - gmesh.ne_b)] = atoi(token[5]);
		        			gmesh.connect[2*(index - gmesh.ne_b) + 1] = atoi(token[6]);
		        			break;
		      			case 2:       //3-point 2D triangle
		      				gmesh.connect[3*(index - gmesh.ne_b)] = atoi(token[5]);
		        			gmesh.connect[3*(index - gmesh.ne_b) + 1] = atoi(token[6]);
		        			gmesh.connect[3*(index - gmesh.ne_b) + 2] = atoi(token[7]);
		        			break;
		      			case 3:       //4-point 2D quadrangle
		      				gmesh.connect[4*(index - gmesh.ne_b)] = atoi(token[5]);
					        gmesh.connect[4*(index - gmesh.ne_b) + 1] = atoi(token[6]);
					        gmesh.connect[4*(index - gmesh.ne_b) + 2] = atoi(token[7]);
					        gmesh.connect[4*(index - gmesh.ne_b) + 3] = atoi(token[8]);
					        break;
		      			case 4:       //4-point 3D tetrahedra
		      				gmesh.connect[4*(index - gmesh.ne_b)] = atoi(token[5]);
					        gmesh.connect[4*(index - gmesh.ne_b) + 1] = atoi(token[6]);
					        gmesh.connect[4*(index - gmesh.ne_b) + 2] = atoi(token[7]);
					        gmesh.connect[4*(index - gmesh.ne_b) + 3] = atoi(token[8]);
					        break;
				      	case 8:       //3-point 1D line
				      		gmesh.connect[3*(index - gmesh.ne_b)] = atoi(token[5]);
					        gmesh.connect[3*(index - gmesh.ne_b) + 1] = atoi(token[6]);
					        gmesh.connect[3*(index - gmesh.ne_b) + 2] = atoi(token[7]);
					        break;
		      			case 9:       //6-point 2D triangle
		      				gmesh.connect[6*(index - gmesh.ne_b)] = atoi(token[5]);
					        gmesh.connect[6*(index - gmesh.ne_b) + 1] = atoi(token[6]);
					        gmesh.connect[6*(index - gmesh.ne_b) + 2] = atoi(token[7]);
					        gmesh.connect[6*(index - gmesh.ne_b) + 3] = atoi(token[8]);
					        gmesh.connect[6*(index - gmesh.ne_b) + 4] = atoi(token[9]);
					        gmesh.connect[6*(index - gmesh.ne_b) + 5] = atoi(token[10]);
					        break;
		      			case 10:      //8-point 2D quadrangle
		      				gmesh.connect[8*(index - gmesh.ne_b)] = atoi(token[5]);
					        gmesh.connect[8*(index - gmesh.ne_b) + 1] = atoi(token[6]);
					        gmesh.connect[8*(index - gmesh.ne_b) + 2] = atoi(token[7]);
					        gmesh.connect[8*(index - gmesh.ne_b) + 3] = atoi(token[8]);
					        gmesh.connect[8*(index - gmesh.ne_b) + 4] = atoi(token[9]);
					        gmesh.connect[8*(index - gmesh.ne_b) + 5] = atoi(token[10]);
					        gmesh.connect[8*(index - gmesh.ne_b) + 6] = atoi(token[11]);
					        gmesh.connect[8*(index - gmesh.ne_b) + 7] = atoi(token[12]);
					        break;
		      			case 11:      //10-point 3D tetrahedra
		      				gmesh.connect[10*(index - gmesh.ne_b)] = atoi(token[5]);
					        gmesh.connect[10*(index - gmesh.ne_b) + 1] = atoi(token[6]);
					        gmesh.connect[10*(index - gmesh.ne_b) + 2] = atoi(token[7]);
					        gmesh.connect[10*(index - gmesh.ne_b) + 3] = atoi(token[8]);
					        gmesh.connect[10*(index - gmesh.ne_b) + 4] = atoi(token[9]);
					        gmesh.connect[10*(index - gmesh.ne_b) + 5] = atoi(token[10]);
					        gmesh.connect[10*(index - gmesh.ne_b) + 6] = atoi(token[11]);
					        gmesh.connect[10*(index - gmesh.ne_b) + 7] = atoi(token[12]);
					        gmesh.connect[10*(index - gmesh.ne_b) + 8] = atoi(token[13]);
					        gmesh.connect[10*(index - gmesh.ne_b) + 9] = atoi(token[14]);
					        break;
		      		}
		    	}
	    		else{
		      		switch (gmesh.type)
		      		{
		      			case 1:       //1-point (2-point 1D line)
		        			gmesh.bconnect[2*index] = atoi(token[3]);
					        gmesh.bconnect[2*index + 1] = atoi(token[5]);
					        break;
		      			case 2:       //2-point line (3-point 2D triangle)
					        gmesh.bconnect[3*index] = atoi(token[3]);
					        gmesh.bconnect[3*index + 1] = atoi(token[5]);
					        gmesh.bconnect[3*index + 2] = atoi(token[6]);
					        break;
		      			case 3:       //2-point line (4-point 2D quadrangle)
		      				gmesh.bconnect[3*index] = atoi(token[3]);
					        gmesh.bconnect[3*index + 1] = atoi(token[5]);
					        gmesh.bconnect[3*index + 2] = atoi(token[6]);
					        break;
		      			case 4:       //3-point triangle (4-point 3D tetrahedra)
		      				gmesh.bconnect[4*index] = atoi(token[3]);
					        gmesh.bconnect[4*index + 1] = atoi(token[5]);
					        gmesh.bconnect[4*index + 2] = atoi(token[6]);
					        gmesh.bconnect[4*index + 3] = atoi(token[7]);
					        break;
		      			case 8:       //1-point (3-point 1D line)
		      				gmesh.bconnect[2*index] = atoi(token[3]);
					        gmesh.bconnect[2*index + 1] = atoi(token[5]);
					        break;
		      			case 9:       //3-point line (6-point 2D triangle)
		      				gmesh.bconnect[4*index] = atoi(token[3]);
					        gmesh.bconnect[4*index + 1] = atoi(token[5]);
					        gmesh.bconnect[4*index + 2] = atoi(token[6]);
					        gmesh.bconnect[4*index + 3] = atoi(token[7]);
					        break;
		      			case 10:      //3-point line (8-point 2D quadrangle)
		      				gmesh.bconnect[4*index] = atoi(token[3]);
					        gmesh.bconnect[4*index + 1] = atoi(token[5]);
					        gmesh.bconnect[4*index + 2] = atoi(token[6]);
					        gmesh.bconnect[4*index + 3] = atoi(token[7]);
					        break;
		      			case 11:      //6-point triangle (10-point 3D tetrahedra)
		      				gmesh.bconnect[7*index] = atoi(token[3]);
					        gmesh.bconnect[7*index + 1] = atoi(token[5]);
					        gmesh.bconnect[7*index + 2] = atoi(token[6]);
					        gmesh.bconnect[7*index + 3] = atoi(token[7]);
					        gmesh.bconnect[7*index + 4] = atoi(token[8]);
					        gmesh.bconnect[7*index + 5] = atoi(token[9]);
					        gmesh.bconnect[7*index + 6] = atoi(token[10]);
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


// bool AssetLoader::load(const std::string& filepath, Font& font)
// {

// }