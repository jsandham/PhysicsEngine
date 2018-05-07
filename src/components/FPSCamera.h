#ifndef __FPSCAMERA_H__
#define __FPSCAMERA_H__

#include "Camera.h"
#include "../Frustum.h"

#include <vector_types.h>

namespace PhysicsEngine
{
	class FPSCamera : public Camera
	{
		private:
			static const GLfloat PAN_SENSITIVITY;
			static const GLfloat SCROLL_SENSITIVITY;
			static const GLfloat TRANSLATE_SENSITIVITY;

			bool firstFrame;

			float yaw, pitch;

			bool jumpInProgress;
			float jumpTime;

			glm::vec2 mouseDelta;

		public:
			FPSCamera();
			FPSCamera(Entity *entity);
			~FPSCamera();

			bool getFirstFrame();
			bool getJumpInProgress();
			float getYaw();
			float getPitch();
			float getJumpTime();

			glm::vec2& getMouseDelta();

			void setFirstFrame(bool firstFrame);
			void setJumpInProgress(bool jumpInProgress);
			void setYaw(float yaw);
			void setPitch(float pitch);
			void setJumpTime(float jumpTime);
			
			void setMouseDelta(glm::vec2& mouseDelta);
	};
}

#endif