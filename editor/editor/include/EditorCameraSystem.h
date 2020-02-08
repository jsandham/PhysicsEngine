#ifndef __EDITOR_CAMERA_SYSTEM_H__
#define __EDITOR_CAMERA_SYSTEM_H__

#include <vector>

#include <systems/System.h>

#include <components/Camera.h>

#include <core/Input.h>

namespace PhysicsEngine
{
	class EditorCameraSystem : public System
	{
	public:
		static const float PAN_SENSITIVITY;
		static const float SCROLL_SENSITIVITY;
		static const float TRANSLATE_SENSITIVITY;

	private:
		Camera* camera;

		int lastPosX;
		int lastPosY;
		int currentPosX;
		int currentPosY;

	public:
		EditorCameraSystem();
		EditorCameraSystem(std::vector<char> data);
		~EditorCameraSystem();

		std::vector<char> serialize();
		void deserialize(std::vector<char> data);

		void init(World* world);
		void update(Input input);

		void resetCamera();
	};

	template<>
	bool IsSystem<EditorCameraSystem>::value = true;
}

#endif
