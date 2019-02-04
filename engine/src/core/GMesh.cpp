#include "../../include/core/PoolAllocator.h"
#include "../../include/core/GMesh.h"

using namespace PhysicsEngine;

GMesh::GMesh()
{

}

GMesh::GMesh(unsigned char* data)
{

}

GMesh::~GMesh()
{

}

void* GMesh::operator new(size_t size)
{
	return getAllocator<GMesh>().allocate();
}

void GMesh::operator delete(void*)
{
	
}