#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Mesh.h"

#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Mesh::Mesh()
{

}

Mesh::Mesh(unsigned char* data)
{

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