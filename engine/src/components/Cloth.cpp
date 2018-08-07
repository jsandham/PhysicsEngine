#include "../../include/components/Cloth.h"
#include "../../include/core/Input.h"

using namespace PhysicsEngine;

Cloth::Cloth()
{
	kappa = 75000.0f;
	c = 1.0f;
	mass = 64.0f / (256 * 256);
}

Cloth::~Cloth()
{
	
}