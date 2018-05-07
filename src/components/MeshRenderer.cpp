#include "MeshRenderer.h"

using namespace PhysicsEngine;

MeshRenderer::MeshRenderer()
{
	visible = true;
}

MeshRenderer::~MeshRenderer()
{

}

bool MeshRenderer::isQueued()
{
	return queued == true;
}

bool MeshRenderer::isVisible()
{
	return visible == true;
}

int MeshRenderer::getMaterialFilter()
{
	return matFilter;
}

int MeshRenderer::getMeshFilter()
{
	return meshFilter;
}

void MeshRenderer::setQueued(bool flag)
{
	queued = flag;
}

void MeshRenderer::setMaterialFilter(int filter)
{
	this->matFilter = filter;
}

void MeshRenderer::setMeshFilter(int filter)
{
	this->meshFilter = filter;
}

Buffer* MeshRenderer::getVertexVBO()
{
	return &vertexVBO;
}

Buffer* MeshRenderer::getNormalVBO()
{
	return &normalVBO;
}

Buffer* MeshRenderer::getTexCoordVBO()
{
	return &texCoordVBO;
}

Buffer* MeshRenderer::getColourVBO()
{
	return &colourVBO;
}

VertexArrayObject* MeshRenderer::getMeshVAO()
{
	return &meshVAO;
}