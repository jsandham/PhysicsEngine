#include "../../include/core/Mesh.h"

#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Mesh::Mesh()
{

}

Mesh::~Mesh()
{

}

void Mesh::apply()
{
	Graphics::apply(this);
}