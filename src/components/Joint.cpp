#include "Joint.h"
#include "Transform.h"

using namespace PhysicsEngine;

Joint::Joint()
{
	connectedJoint = NULL;
	connectedBody = NULL;
}

Joint::~Joint()
{
	
}

Joint* Joint::getConnectedJoint()
{
	return connectedJoint;
}

Rigidbody* Joint::getConnectedBody()
{
	return connectedBody;
}

glm::vec3 Joint::getConnectedAnchor()
{
	if(connectedJoint != NULL){
		return connectedJoint->getAnchor();
	}

	return glm::vec3(0.0f, 5.0f, 0.0f);
}

glm::vec3 Joint::getAnchor()
{
	glm::vec3 position = entity->getComponent<Transform>()->position;

	return position + anchor;
}

void Joint::setConnectedJoint(Joint* joint)
{
	this->connectedJoint = joint;
}

void Joint::setConnectedBody(Rigidbody* body)
{
	this->connectedBody = body;
}

void Joint::setConnectedAnchor(glm::vec3 anchor)
{
	this->connectedAnchor = anchor;
}

void Joint::setAnchor(glm::vec3 anchor)
{
	this->anchor = anchor;
}