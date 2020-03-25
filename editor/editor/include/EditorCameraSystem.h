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
		Camera* mCamera;

		int mLastPosX;
		int mLastPosY;
		int mCurrentPosX;
		int mCurrentPosY;

	public:
		EditorCameraSystem();
		EditorCameraSystem(std::vector<char> data);
		~EditorCameraSystem();

		std::vector<char> serialize() const;
		std::vector<char> serialize(Guid systemId) const;
		void deserialize(std::vector<char> data);

		void init(World* world);
		void update(Input input);

		void resetCamera();
	};

	template <>
	const int SystemType<EditorCameraSystem>::type = 21;

	template< typename T>
	struct IsEditorCameraSystem { static const bool value; };

	template<typename T>
	const bool IsEditorCameraSystem<T>::value = false;

	template<>
	const bool IsEditorCameraSystem<EditorCameraSystem>::value = true;
	template<>
	const bool IsSystem<EditorCameraSystem>::value = true;
}

#endif
