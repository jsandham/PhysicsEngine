#ifndef __CAMERA_H__
#define __CAMERA_H__

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

#include "../core/Frustum.h"

namespace PhysicsEngine
{
	enum class CameraMode
	{
		Main,
		Secondary
	};

	enum class CameraSSAO
	{
		SSAO_On,
		SSAO_Off,
	};

#pragma pack(push, 1)
	struct CameraHeader
	{
		Guid mComponentId;
		Guid mEntityId;
		Guid mTargetTextureId;
		CameraMode mMode;
		CameraSSAO mSSAO;
		glm::vec4 mBackgroundColor;
		int mX;
		int mY;
		int mWidth;
		int mHeight;
		float mFov;
		float mAspectRatio;
		float mNearPlane;
		float mFarPlane;
	};
#pragma pack(pop)

	struct Viewport
	{
		int mX;
		int mY;
		int mWidth;
		int mHeight;
	};

	class Camera : public Component
	{
		public:
			Frustum mFrustum;
			Viewport mViewport;
			Guid mTargetTextureId;

			CameraMode mMode;
			CameraSSAO mSSAO;

			glm::vec4 mBackgroundColor;

		private:
			GLuint mMainFBO;
			GLuint mColorTex;
			GLuint mDepthTex;

			GLuint mGeometryFBO;
			GLuint mPositionTex;
			GLuint mNormalTex;

			GLuint mSsaoFBO;
			GLuint mSsaoColorTex;
			GLuint mSsaoNoiseTex;

			glm::vec3 mSsaoSamples[64];
			glm::mat4 viewMatrix;

			bool mIsCreated;

		public:
			Camera();
			Camera(std::vector<char> data);
			~Camera();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(std::vector<char> data);

			void create();
			void destroy();
			void computeViewMatrix(glm::vec3 position, glm::vec3 forward, glm::vec3 up);

			bool isCreated() const;
			glm::mat4 getViewMatrix() const;
			glm::mat4 getProjMatrix() const;
			glm::vec3 getSSAOSample(int sample) const;

			GLuint getNativeGraphicsMainFBO() const;
			GLuint getNativeGraphicsGeometryFBO() const;
			GLuint getNativeGraphicsSSAOFBO() const;
			GLuint getNativeGraphicsColorTex() const;
			GLuint getNativeGraphicsDepthTex() const;
			GLuint getNativeGraphicsPositionTex() const;
			GLuint getNativeGraphicsNormalTex() const;
			GLuint getNativeGraphicsSSAOColorTex() const;
			GLuint getNativeGraphicsSSAONoiseTex() const;
	};

	template <>
	const int ComponentType<Camera>::type = 2;

	template <typename T>
	struct IsCamera { static const bool value; };

	template <typename T>
	const bool IsCamera<T>::value = false;

	template<>
	const bool IsCamera<Camera>::value = true;
	template<>
	const bool IsComponent<Camera>::value = true;
}

#endif