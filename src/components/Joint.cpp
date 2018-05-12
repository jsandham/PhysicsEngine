#include "Joint.h"

using namespace PhysicsEngine;

Joint::Joint()
{
	connectedJoint = NULL;
	connectedBody = NULL;
}

Joint::~Joint()
{
	
}

Rigidbody* Joint::getConnectedBody()
{
	return connectedBody;
}

glm::vec3 Joint::getConnectedAnchor()
{
	if(connectedJoint != NULL){
		connectedAnchor = connectedJoint->getAnchor(); 
	}

	return connectedAnchor; 
}

glm::vec3 Joint::getAnchor()
{
	return anchor;
}

void Joint::setConnectedBody(Rigidbody* body)
{
	this->connectedBody = body;

	connectedJoint = connectedBody->entity->getComponent<Joint>();

	if(connectedJoint != NULL){
		connectedAnchor = connectedJoint->getAnchor();
	}
}

void Joint::setConnectedAnchor(glm::vec3 anchor)
{
	this->connectedAnchor = anchor;
}

void Joint::setAnchor(glm::vec3 anchor)
{
	this->anchor = anchor;
}