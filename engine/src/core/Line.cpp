#include "../../include/core/Line.h"

using namespace PhysicsEngine;

Line::Line()
{

}

Line::Line(glm::vec3 start, glm::vec3 end)
{
	this->start = start;
	this->end = end;
}

Line::~Line()
{

}