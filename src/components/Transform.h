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
		private:
			glm::vec3 position;
			glm::vec3 eulerAngles;
			glm::quat rotation;
			glm::vec3 scale;

			glm::mat4 translateMatrix;
			glm::mat4 scaleMatrix;
			glm::mat4 rotationMatrix;
			glm::mat4 modelMatrix;

		public:
			Transform();
			Transform(Entity *entity);
			~Transform();

			glm::vec3 getPosition();
			glm::vec3 getEulerAngles();
			glm::quat getRotation();
			glm::vec3 getScale();

			glm::mat4 getModelMatrix();

			void setPosition(glm::vec3 position);
			void setEulerAngles(glm::vec3 eulerAngles);
			void setRotation(glm::quat rotation);
			void setScale(glm::vec3 scale);
	};
}

#endif