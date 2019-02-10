#include <iostream>

#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Mesh.h"

#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Mesh::Mesh()
{

}

Mesh::Mesh(std::vector<char> data)
{
	size_t index = sizeof(int);
	MeshHeader* header = reinterpret_cast<MeshHeader*>(&data[index]);

	assetId = header->meshId;
	vertices.resize(header->verticesSize);
	normals.resize(header->normalsSize);
	texCoords.resize(header->texCoordsSize);

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

	std::cout << "mesh index: " << index << " data size: " << data.size() << std::endl;
}

Mesh::~Mesh()
{

}

void* Mesh::operator new(size_t size)
{
	return getAllocator<Mesh>().allocate();
}

void Mesh::operator delete(void*)
{
	
}

void Mesh::apply()
{
	Graphics::apply(this);
}