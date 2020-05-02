#ifndef __EDITOR_CAMERA_SYSTEM_H__
#define __EDITOR_CAMERA_SYSTEM_H__

#include <vector>

#include <systems/System.h>

#include <components/Transform.h>
#include <components/Camera.h>

#include <core/Input.h>

namespace PhysicsEngine
{
	class EditorCameraSystem : public System
	{
	private:
		static const float YAW_PAN_SENSITIVITY;
		static const float PITCH_PAN_SENSITIVITY;
		static const float ZOOM_SENSITIVITY;
		static const float TRANSLATE_SENSITIVITY;

	private:
		Transform* mTransform;
		Camera* mCamera;

		int mMousePosX;
		int mMousePosY;
		int mMousePosXOnRightClick;
		int mMousePosYOnRightClick;
		glm::quat rotationOnClick;

	public:
		EditorCameraSystem();
		EditorCameraSystem(std::vector<char> data);
		~EditorCameraSystem();

		std::vector<char> serialize() const;
		std::vector<char> serialize(Guid systemId) const;
		void deserialize(std::vector<char> data);

		void init(World* world);
		void update(Input input, Time time);

		void resetCamera();
		void setViewport(Viewport viewport);
		void setFrustum(Frustum frustum);

		Viewport getViewport() const;
		Frustum getFrustum() const;

		Guid getMeshRendererUnderMouse(float nx, float ny) const;
		int getMousePosX() const;
		int getMousePosY() const;
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
