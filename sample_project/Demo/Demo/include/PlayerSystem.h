#ifndef __PLAYERSYSTEM_H__
#define __PLAYERSYSTEM_H__

#include <systems/System.h>

#include <components/Camera.h>

#include <core/Input.h>

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
		PlayerSystem();
		PlayerSystem(unsigned char* data);
		~PlayerSystem();

		void init();
		void update(Input input);
	};
}

#endif