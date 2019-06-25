#include <iostream>

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Mesh.h"

using namespace PhysicsEngine;

Mesh::Mesh()
{
	assetId = Guid::INVALID;
}

Mesh::Mesh(std::vector<char> data)
{
	size_t index = sizeof(int);
	MeshHeader* header = reinterpret_cast<MeshHeader*>(&data[index]);

	assetId = header->meshId;
	vertices.resize(header->verticesSize);
	normals.resize(header->normalsSize);
	texCoords.resize(header->texCoordsSize);
	subMeshStartIndicies.resize(header->subMeshStartIndiciesSize);

	index += sizeof(MeshHeader);

	for(size_t i = 0; i < header->verticesSize; i++){
		vertices[i] = *reinterpret_cast<float*>(&data[index + sizeof(float) * i]);
	}

	index += vertices.size() * sizeof(float);

	for(size_t i = 0; i < header->normalsSize; i++){
		normals[i] = *reinterpret_cast<float*>(&data[index + sizeof(float) * i]);
	}

	index += normals.size() * sizeof(float);

	for(size_t i = 0; i < header->texCoordsSize; i++){
		texCoords[i] = *reinterpret_cast<float*>(&data[index + sizeof(float) * i]);
	}

	index += texCoords.size() * sizeof(float);

	for(size_t i = 0; i < header->subMeshStartIndiciesSize; i++){
		subMeshStartIndicies[i] = *reinterpret_cast<int*>(&data[index + sizeof(int) * i]);
	}

	index += subMeshStartIndicies.size() * sizeof(int);

	std::cout << "mesh index: " << index << " data size: " << data.size() << std::endl;
}

Mesh::~Mesh()
{

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