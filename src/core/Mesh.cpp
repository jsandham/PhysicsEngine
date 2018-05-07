#include "Mesh.h"

using namespace PhysicsEngine;

Mesh::Mesh()
{

}

Mesh::~Mesh()
{

}

std::vector<float>& Mesh::getVertices()
{
	return vertices;
}

std::vector<float>& Mesh::getNormals()
{
	return normals;
}

std::vector<float>& Mesh::getTexCoords()
{
	return texCoords;
}

std::vector<float>& Mesh::getColours()
{
	return colours;
}

void Mesh::setVertices(std::vector<float> &vertices)
{
	this->vertices = vertices;
}

void Mesh::setNormals(std::vector<float> &normals)
{
	this->normals = normals;
}

void Mesh::setTexCoords(std::vector<float> &texCoords)
{
	this->texCoords = texCoords;
}

void Mesh::setColours(std::vector<float>& colours)
{
	this->colours = colours;
}