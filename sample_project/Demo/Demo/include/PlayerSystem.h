#ifndef __PLAYERSYSTEM_H__
#define __PLAYERSYSTEM_H__

#include <vector>

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
		PlayerSystem(std::vector<char> data);
		~PlayerSystem();

		void* operator new(size_t size);
		void operator delete(void*);

		void init(World* world);
		void update(Input input);
	};
}

#endif