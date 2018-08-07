#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

#include "Component.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtx/quaternion.hpp"
#include "../glm/gtc/matrix_transform.hpp"

namespace PhysicsEngine
{
	class Transform : public Component
	{
		public:
			glm::vec3 position;
			glm::quat rotation;
			glm::vec3 scale;

		public:
			Transform();
			~Transform();

			glm::vec3 getEulerAngles();
			glm::mat4 getModelMatrix();

			void setEulerAngles(glm::vec3 eulerAngles);
	};
}

#endif