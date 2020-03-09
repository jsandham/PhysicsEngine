#include <iostream>

#include "../../include/core/Log.h"
#include "../../include/core/Mesh.h"
#include "../../include/obj_load/obj_load.h"

using namespace PhysicsEngine;

Mesh::Mesh()
{
	assetId = Guid::INVALID;
	this->isCreated = false;
}

Mesh::Mesh(std::vector<char> data)
{
	deserialize(data);
}

Mesh::~Mesh()
{

}

std::vector<char> Mesh::serialize() const
{
	return serialize(assetId);
}

std::vector<char> Mesh::serialize(Guid assetId) const
{
	MeshHeader header;
	header.meshId = assetId;
	header.verticesSize = vertices.size();
	header.normalsSize = normals.size();
	header.texCoordsSize = texCoords.size();
	header.subMeshVertexStartIndiciesSize = subMeshVertexStartIndices.size();

	size_t numberOfBytes = sizeof(MeshHeader) +
		vertices.size() * sizeof(float) +
		normals.size() * sizeof(float) +
		texCoords.size() * sizeof(float) +
		subMeshVertexStartIndices.size() * sizeof(int);

	std::vector<char> data(numberOfBytes);

	size_t start1 = 0;
	size_t start2 = start1 + sizeof(MeshHeader);
	size_t start3 = start2 + sizeof(float) * vertices.size();
	size_t start4 = start3 + sizeof(float) * normals.size();
	size_t start5 = start4 + sizeof(float) * texCoords.size();

	memcpy(&data[start1], &header, sizeof(MeshHeader));
	memcpy(&data[start2], &vertices[0], sizeof(float) * vertices.size());
	memcpy(&data[start3], &normals[0], sizeof(float) * normals.size());
	memcpy(&data[start4], &texCoords[0], sizeof(float) * texCoords.size());
	memcpy(&data[start5], &subMeshVertexStartIndices[0], sizeof(int) * subMeshVertexStartIndices.size());

	return data;
}

void Mesh::deserialize(std::vector<char> data)
{
	size_t start1 = 0;
	size_t start2 = start1 + sizeof(MeshHeader);

	MeshHeader* header = reinterpret_cast<MeshHeader*>(&data[start1]);

	assetId = header->meshId;
	vertices.resize(header->verticesSize);
	normals.resize(header->normalsSize);
	texCoords.resize(header->texCoordsSize);
	subMeshVertexStartIndices.resize(header->subMeshVertexStartIndiciesSize);

	size_t start3 = start2 + sizeof(float) * vertices.size();
	size_t start4 = start3 + sizeof(float) * normals.size();
	size_t start5 = start4 + sizeof(float) * texCoords.size();

	for(size_t i = 0; i < header->verticesSize; i++){
		vertices[i] = *reinterpret_cast<float*>(&data[start2 + sizeof(float) * i]);
	}

	for(size_t i = 0; i < header->normalsSize; i++){
		normals[i] = *reinterpret_cast<float*>(&data[start3 + sizeof(float) * i]);
	}

	for(size_t i = 0; i < header->texCoordsSize; i++){
		texCoords[i] = *reinterpret_cast<float*>(&data[start4 + sizeof(float) * i]);
	}

	for(size_t i = 0; i < header->subMeshVertexStartIndiciesSize; i++){
		subMeshVertexStartIndices[i] = *reinterpret_cast<int*>(&data[start5 + sizeof(int) * i]);
	}

	this->isCreated = false;
}

void Mesh::load(const std::string& filepath)
{
	obj_mesh mesh;
	
	if (obj_load(filepath, mesh))
	{
		vertices = mesh.vertices;
		normals = mesh.normals;
		texCoords = mesh.texCoords;
		subMeshVertexStartIndices = mesh.subMeshVertexStartIndices;
	}
	else {
		std::string message = "Error: Could not load obj mesh " + filepath + "\n";
		Log::error(message.c_str());
	}
}

void Mesh::load(std::vector<float> vertices, std::vector<float> normals, std::vector<float> texCoords, std::vector<int> subMeshStartIndices)
{
	this->vertices = vertices;
	this->normals = normals;
	this->texCoords = texCoords;
	this->subMeshVertexStartIndices = subMeshStartIndices;
}

const std::vector<float>& Mesh::getVertices() const
{
	return vertices;
}

const std::vector<float>& Mesh::getNormals() const
{
	return normals;
}

const std::vector<float>& Mesh::getTexCoords() const
{
	return texCoords;
}

const std::vector<int>& Mesh::getSubMeshStartIndices() const
{
	return subMeshVertexStartIndices;
}

Sphere Mesh::getBoundingSphere() const
{
	// Ritter algorithm for bounding sphere
	// find furthest point from first vertex
	glm::vec3 x = glm::vec3(vertices[0], vertices[1], vertices[2]);
	glm::vec3 y = x;
	float maxDistance = 0.0f;
	for(size_t i = 1; i < vertices.size() / 3; i++){
		
		glm::vec3 temp = glm::vec3(vertices[3*i], vertices[3*i + 1], vertices[3*i + 2]);
		float distance = glm::distance(x, temp);
		if(distance > maxDistance){
			y = temp;
			distance = maxDistance;
		}
	}

	// now find furthest point from y
	glm::vec3 z = y;
	maxDistance = 0.0f;
	for(size_t i = 0; i < vertices.size() / 3; i++){

		glm::vec3 temp = glm::vec3(vertices[3*i], vertices[3*i + 1], vertices[3*i + 2]);
		float distance = glm::distance(y, temp);
		if(distance > maxDistance){
			z = temp;
			distance = maxDistance;
		}
	}

	Sphere sphere;
	sphere.radius = glm::distance(y, z);
	sphere.centre = 0.5f * (y + z);

	for(size_t i = 0; i < vertices.size() / 3; i++){
		glm::vec3 temp = glm::vec3(vertices[3*i], vertices[3*i + 1], vertices[3*i + 2]);
		float distance = glm::distance(temp, sphere.centre);
		if(distance > sphere.radius){
			sphere.radius += distance;
		}
	}

	std::cout << "centre: " << sphere.centre.x << " " << sphere.centre.y << " " << sphere.centre.z << " radius: " << sphere.radius << std::endl;

	return sphere;
}