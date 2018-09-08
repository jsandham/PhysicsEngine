#include "../../include/core/Mesh.h"

#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Mesh::Mesh()
{
	meshId = -1;
	//globalIndex = -1;
}

Mesh::~Mesh()
{

}

void Mesh::apply()
{
	Graphics::apply(this);
}