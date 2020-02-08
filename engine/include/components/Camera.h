#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#undef NEAR
#undef FAR
#undef near
#undef far

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"
#include "../glm/gtc/type_ptr.hpp"

#include "Component.h"

#include "../core/Guid.h"

namespace PhysicsEngine
{
	enum class CameraMode
	{
		Main,
		Secondary
	};

#pragma pack(push, 1)
	struct CameraHeader
	{
		Guid componentId;
		Guid entityId;
		Guid targetTextureId;
		CameraMode mode;
		glm::vec3 position;
		glm::vec3 front;
		glm::vec3 up;
		glm::vec4 backgroundColor;
		int x;
		int y;
		int width;
		int height;
		float fov;
		float nearPlane;
		float farPlane;
	};
#pragma pack(pop)

	// plane defined by n.x*x + n.y*y + n.z*z + d = 0, where d = -dot(n, x0)
	struct Plane
	{
		glm::vec3 n;
		glm::vec3 x0;

		float distance(glm::vec3 point) const;
	};

	struct Viewport
	{
		int x;
		int y;
		int width;
		int height;

		float getAspectRatio() const;
	};

	struct Frustum
	{
		Plane planes[6];

		float fov;
		float nearPlane;
		float farPlane;

		int checkPoint(glm::vec3 point) const;
		int checkSphere(glm::vec3 centre, float radius) const;
		int checkAABB(glm::vec3 min, glm::vec3 max) const;
	};

	class Camera : public Component
	{
		public:
			Frustum frustum;
			Viewport viewport;
			Guid targetTextureId;
			
			GLuint mainFBO;
			GLuint colorTex;
			GLuint depthTex;

			GLuint geometryFBO;
			GLuint positionTex;
			GLuint normalTex;

			GLuint ssaoFBO;
			GLuint ssaoColorTex;
		
			GLuint ssaoNoiseTex;

			std::vector<glm::vec3> ssaoSamples;

			enum {
				TOP = 0,
				BOTTOM,
				LEFT,
				RIGHT,
				NEAR,
				FAR
			};

			CameraMode mode;

			glm::vec3 position;
			glm::vec3 front;
			glm::vec3 up;
			glm::vec3 right;
			glm::vec4 backgroundColor;

			bool isCreated;
			bool useSSAO;

		public:
			Camera();
			Camera(std::vector<char> data);
			~Camera();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			void updateInternalCameraState();

			glm::mat4 getViewMatrix() const;
			glm::mat4 getProjMatrix() const;

			int checkPointInFrustum(glm::vec3 point) const;
			int checkSphereInFrustum(glm::vec3 centre, float radius) const;
			int checkAABBInFrustum(glm::vec3 min, glm::vec3 max) const;
	};

	template <>
	const int ComponentType<Camera>::type = 2;

	template <typename T>
	struct IsCamera { static bool value; };

	template <typename T>
	bool IsCamera<T>::value = false;

	template<>
	bool IsCamera<Camera>::value = true;
	template<>
	bool IsComponent<Camera>::value = true;
}

#endif