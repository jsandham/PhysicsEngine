#include "MeshRenderer.h"

using namespace PhysicsEngine;

MeshRenderer::MeshRenderer()
{
	meshFilter = -1;
	materialFilter = -1;
}

MeshRenderer::MeshRenderer(Entity* entity)
{
	this->entity = entity;

	meshFilter = -1;
	materialFilter = -1;
}

MeshRenderer::~MeshRenderer()
{
}