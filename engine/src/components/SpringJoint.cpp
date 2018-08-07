#include "../../include/components/SpringJoint.h"

using namespace PhysicsEngine;

SpringJoint::SpringJoint()
{
	damping = 0.0f;
	stiffness = 0.1f;
	restLength = 1.0f;
	minDistance = 0.0f;
	maxDistance = 0.0f;

	connectedAnchor = glm::vec3(0.0f, 1.0f, 0.0f);
	anchor = glm::vec3(0.0f, 0.0f, 0.0f);
}

SpringJoint::~SpringJoint()
{
	
}

glm::vec3 SpringJoint::getTargetPosition()
{
	glm::vec3 ca = getConnectedAnchor();
	glm::vec3 a = getAnchor();

	return ca - restLength * glm::normalize(ca - a);
}