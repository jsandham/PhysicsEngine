#include "GMesh.h"

using namespace PhysicsEngine;

GMesh::GMesh()
{

}

GMesh::~GMesh()
{

}

std::vector<float>& GMesh::getVertices()
{
	return vertices;
}

std::vector<int>& GMesh::getConnect()
{
	return connect;
}

std::vector<int>& GMesh::getBConnect()
{
	return bconnect;
}

std::vector<int>& GMesh::getGroups()
{
	return groups;
}

void GMesh::setVertices(std::vector<float> &vertices)
{
	this->vertices = vertices;
}

void GMesh::setConnect(std::vector<int> &connect)
{
	this->connect = connect;
}

void GMesh::setBConnect(std::vector<int> &bconnect)
{
	this->bconnect = bconnect;
}

void GMesh::setGroups(std::vector<int> &groups)
{
	this->groups = groups;
}