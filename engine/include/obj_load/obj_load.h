#ifndef OBJ_LOAD_H__
#define OBJ_LOAD_H__

#include <vector>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>

// Simple obj mesh header only code. Replace with some other mesh loading library (preferrable header only)

typedef struct obj_mesh
{
	std::vector<float> mVertices;
	std::vector<float> mNormals;
	std::vector<float> mTexCoords;
	std::vector<int> mSubMeshVertexStartIndices;
}obj_mesh;

void obj_split(const std::string& s, char delim, std::vector<std::string>& elems) {
	std::stringstream ss(s);
	std::string item;
	while (getline(ss, item, delim)) {
		if (item.length() > 0) {
			elems.push_back(item);
		}
	}
}

std::vector<std::string> obj_split(const std::string& s, char delim) {
	std::vector<std::string> elems;
	elems.reserve(10);
	obj_split(s, delim, elems);
	return elems;
}

bool obj_load(const std::string& filepath, obj_mesh& mesh)
{
	std::ifstream file;
	file.open(filepath, std::ios::in);

	if (!file.is_open()) {
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
	while (!file.eof()) {
		lineNumber++;
		getline(file, line);

		std::vector< std::string > words = obj_split(line, ' ');

		if (words.size() == 0) { continue; }

		size_t wordsSize = words.size();

		std::string command = words[0];

		if (command == "v") {
			if (!(wordsSize - 1 == 3 || wordsSize - 1 == 4)) {
				error = true;
				errorString = "vertex at line number " + std::to_string(lineNumber) + " must contain x y z (w) components but has " + std::to_string(wordsSize);
			}

			for (size_t i = 1; i < words.size(); i++) {
				float number;
				std::stringstream(words[i]) >> number;
				v.push_back(number);
			}

			if (wordsSize - 1 == 3) {
				v.push_back(1.0);
			}
		}
		else if (command == "vn") {
			if (!(wordsSize - 1 == 3)) {
				error = true;
				errorString = "normal at line number " + std::to_string(lineNumber) + " must contain exactly i j k components";
			}

			for (size_t i = 1; i < words.size(); i++) {
				float number;
				std::stringstream(words[i]) >> number;
				vn.push_back(number);
			}
		}
		else if (command == "vt") {
			if (!(wordsSize - 1 == 1 || wordsSize - 1 == 2 || wordsSize - 1 == 3)) {
				error = true;
				errorString = "texture coordinate at line number " + std::to_string(lineNumber) + " must contain u (v) (w) components";
			}

			for (size_t i = 1; i < words.size(); i++) {
				float number;
				std::stringstream(words[i]) >> number;
				vt.push_back(number);
			}

			for (size_t i = words.size(); i < 4; i++) {
				vt.push_back(0.0f);
			}
		}
		else if (command == "f") {
			faceCount++;
			faceStartIndices.push_back(faceStartIndices.back() + (int)words.size() - 1);

			for (size_t i = 1; i < words.size(); i++) {
				std::string word = words[i];

				int firstSlashIndex = -1;
				int secondSlashIndex = -1;
				size_t index = 0;
				while (index < word.size()) {
					if (word[index] == '/') {
						firstSlashIndex = (int)index;
						break;
					}

					index++;
				}

				index++;
				while (index < word.size()) {
					if (word[index] == '/') {
						secondSlashIndex = (int)index;
						break;
					}

					index++;
				}

				if (firstSlashIndex == -1 && secondSlashIndex != -1) {
					error = true;
					errorString = "Error: Could not find first slash index but could find second when parsing faces";
				}

				std::string vStr = word;
				std::string vtStr = "";
				std::string vnStr = "";

				//std::cout << "first slash index: " << firstSlashIndex << " second slash index: " << secondSlashIndex << std::endl;

				if (firstSlashIndex != -1 && secondSlashIndex != -1 /*2 slashes exist*/) {
					vStr = word.substr(0, firstSlashIndex - 0);
					if (secondSlashIndex - firstSlashIndex > 1) {
						vtStr = word.substr(firstSlashIndex + 1, secondSlashIndex - firstSlashIndex - 1);
					}
					vnStr = word.substr(secondSlashIndex + 1, word.size() - secondSlashIndex - 1);
				}
				else if (firstSlashIndex != -1 && secondSlashIndex == -1 /*1 slash exists*/) {
					vStr = word.substr(0, firstSlashIndex - 0);
					vtStr = word.substr(firstSlashIndex + 1, secondSlashIndex - firstSlashIndex - 1);
				}

				//TODO: Check that vStr, vtStr, and vnStr are valid numbers
				int number;
				std::istringstream(vStr) >> number;
				f_v.push_back(number);
				if (vtStr.length() > 0) {
					std::istringstream(vtStr) >> number;
					f_vt.push_back(number);
				}
				if (vnStr.length() > 0) {
					std::istringstream(vnStr) >> number;
					f_vn.push_back(number);
				}
			}

			if (f_vt.size() != f_v.size() && f_vt.size() > 0) {
				error = true;
				errorString = "Error: Incorrect number of texture coordinates found in faces";
			}
			if (f_vn.size() != f_v.size() && f_vn.size() > 0) {
				error = true;
				errorString = "Error: Incorrect number of normals found in faces";
			}
		}
		else if (command == "g") {
		}
		else if (command == "usemtl") {
			subMeshFaceStartIndices.push_back(faceCount);
		}

		if (error) {
			break;
		}
	}

	subMeshFaceStartIndices.push_back(faceCount);

	// calculate normals if not given
	if (vn.size() == 0) {
		vn.resize(3 * (v.size() / 4), 0.0f); // v can have 4 components per vertex by the obj standard
		f_vn.resize(f_v.size(), 0);

		for (size_t i = 0; i < f_v.size(); i += 3) {
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
			float cLength = sqrt(c_x * c_x + c_y * c_y + c_z * c_z);
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

		for (size_t i = 0; i < vn.size(); i += 3) {
			float v_x = vn[i];
			float v_y = vn[i + 1];
			float v_z = vn[i + 2];

			float vLength = sqrt(v_x * v_x + v_y * v_y + v_z * v_z);
			vn[i] = vn[i] / vLength;
			vn[i + 1] = vn[i + 1] / vLength;
			vn[i + 2] = vn[i + 2] / vLength;
		}
	}

	std::vector<float> vertices;
	std::vector<float> normals;
	std::vector<float> texCoords;
	std::vector<int> subMeshVertexStartIndices;

	// std::cout << "f_vt: " << std::endl;
	// for(size_t i = 0; i < 20; i++){
	// 	std::cout << f_vt[i] << " ";
	// }

	// loop through each sub mesh
	for (size_t i = 0; i < subMeshFaceStartIndices.size() - 1; i++) {
		int subMeshFaceStartIndex = subMeshFaceStartIndices[i];
		int subMeshFaceEndIndex = subMeshFaceStartIndices[i + 1];

		subMeshVertexStartIndices.push_back((int)vertices.size());

		// loop through all faces in sub mesh
		for (size_t j = subMeshFaceStartIndex; j < subMeshFaceEndIndex; j++) {
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

			if (endIndex - startIndex == 3 /*triangle face*/) {
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

				if (f_vt.size() > 0) {
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

				if (f_vn.size() > 0) {
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
			else if (endIndex - startIndex == 4 /*quadrilateral face*/) {
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

				if (f_vt.size() > 0) {
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

				if (f_vn.size() > 0) {
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
			else {
				error = true;
				errorString = "Error: face (" + std::to_string(faceCount) + ") with " + std::to_string(endIndex - startIndex) + " vertices not currently supported";
			}
		}
	}

	subMeshVertexStartIndices.push_back((int)vertices.size());

	mesh.mVertices = vertices;
	mesh.mNormals = normals;
	mesh.mTexCoords = texCoords;
	mesh.mSubMeshVertexStartIndices = subMeshVertexStartIndices;

	if (error) {
		std::cout << "Error: " << errorString << std::endl;
		return false;
	}

	std::cout << "done" << std::endl;

	file.close();

	return true;
}


#endif