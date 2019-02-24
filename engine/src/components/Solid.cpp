#include "../../include/components/Solid.h"

#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

Solid::Solid()
{
	
}

Solid::Solid(std::vector<char> data)
{
	
}

Solid::~Solid()
{
	
}

void* Solid::operator new(size_t size)
{
	return getAllocator<Solid>().allocate();
}

void Solid::operator delete(void*)
{

}