#include "SpringJoint.h"

using namespace PhysicsEngine;

SpringJoint::SpringJoint()
{
	damping = 0.3f;
	stiffness = 1.0f;
	restLength = 1.0f;
	minDistance = 0.0f;
	maxDistance = 0.5f;

	connectedAnchor = glm::vec3(0.0f, 1.0f, 0.0f);
	anchor = glm::vec3(0.0f, 0.0f, 0.0f);
}

SpringJoint::SpringJoint(Entity* entity)
{
	this->entity = entity;

	damping = 0.3f;
	stiffness = 1.0f;
	restLength = 1.0f;
	minDistance = 0.0f;
	maxDistance = 0.5f;

	connectedAnchor = glm::vec3(0.0f, 1.0f, 0.0f);
	anchor = glm::vec3(0.0f, 0.0f, 0.0f);
}

SpringJoint::~SpringJoint()
{
	
}

glm::vec3 SpringJoint::getTargetPosition()
{
	//connectedAnchor = getConnectedAnchor();

	return connectedAnchor + 0.5f * restLength * glm::normalize(connectedAnchor - anchor);
}