#ifndef __HINGE_JOINT_H__
#define __HINGE_JOINT_H__

#include "Joint.h"

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class HingeJoint : public Joint
	{
		public:
			glm::vec3 axis;
			float targetAngle;
			float stiffness;
			float damper;

		public:
			HingeJoint();
			~HingeJoint();
	};
}

#endif