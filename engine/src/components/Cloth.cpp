#include "../../include/components/Cloth.h"

#include "../../include/core/Input.h"
#include "../../include/core/PoolAllocator.h"

using namespace PhysicsEngine;

Cloth::Cloth()
{
	kappa = 75000.0f;
	c = 1.0f;
	mass = 64.0f / (256 * 256);
}

Cloth::Cloth(std::vector<char> data)
{
	
}

Cloth::~Cloth()
{
	
}

void* Cloth::operator new(size_t size)
{
	return getAllocator<Cloth>().allocate();
}

void Cloth::operator delete(void*)
{

}