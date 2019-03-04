#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <iostream>
#include <vector>

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"
#include "../glm/gtc/type_ptr.hpp"

#include "Component.h"
#include "../core/Frustum.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct CameraHeader
	{
		Guid componentId;
		Guid entityId;
		glm::vec3 position;
		glm::vec4 backgroundColor;
	};
#pragma pack(pop)

	class Camera : public Component
	{
		public:
			Frustum frustum;

			glm::vec3 position;
			glm::vec3 front;
			glm::vec3 up;
			glm::vec3 right;
			glm::vec3 worldUp;

			glm::mat4 projection;
			glm::mat4 view;
			glm::vec4 backgroundColor;

		public:
			bool enabled;
			int priority;
			int x, y;
			int width, height;

			int lastPosX;
			int lastPosY;
			int currentPosX;
			int currentPosY;

		public:
			Camera();
			Camera(std::vector<char> data);
			~Camera();

			glm::vec3& getPosition();
			glm::vec3& getFront();
			glm::vec3& getUp();
			glm::vec3& getRight();
			glm::vec3& getWorldUp();
			//int2& getLastPosition();
			//int2& getCurrentPosition();
			glm::mat4& getViewMatrix();
			glm::mat4& getProjMatrix();
			glm::vec4& getBackgroundColor();

			void setPosition(glm::vec3& position);
			void setFront(glm::vec3& front);
			void setUp(glm::vec3& up);
			void setRight(glm::vec3& right);
			//void setLastPosition(int2& lastPos);
			//void setCurrentPosition(int2& currentPos);
			void setProjMatrix(glm::mat4& projection);
			void setBackgroundColor(glm::vec4& backgroundColor);

			int checkPointInFrustum(glm::vec3 point);
			int checkSphereInFrustum(glm::vec3 centre, float radius);
			int checkAABBInFrustum(glm::vec3 min, glm::vec3 max);
	};

}


#endif