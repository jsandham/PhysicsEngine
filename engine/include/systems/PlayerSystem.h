#ifndef __FPSCAMERASYSTEM_H__
#define __FPSCAMERASYSTEM_H__

#include "System.h"

#include "../components/Camera.h"

namespace PhysicsEngine
{
	class PlayerSystem : public System
	{
		public:
			static const float PAN_SENSITIVITY;
			static const float SCROLL_SENSITIVITY;
			static const float TRANSLATE_SENSITIVITY;

		private:
			Camera* camera;

		public:
			// PlayerSystem(Manager *manager, SceneContext* context);
			PlayerSystem();
			PlayerSystem(unsigned char* data);
			~PlayerSystem();

			void init();
			void update();
	};
}

#endif