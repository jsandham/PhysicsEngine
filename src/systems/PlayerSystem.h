#ifndef __FPSCAMERASYSTEM_H__
#define __FPSCAMERASYSTEM_H__

#include "System.h"

#include "../components/Camera.h"

namespace PhysicsEngine
{
	class PlayerSystem : public System
	{
		public:
			static const GLfloat PAN_SENSITIVITY;
			static const GLfloat SCROLL_SENSITIVITY;
			static const GLfloat TRANSLATE_SENSITIVITY;

		private:
			Camera* camera;

		public:
			PlayerSystem(Manager *manager);
			~PlayerSystem();

			void init();
			void update();
	};
}

#endif