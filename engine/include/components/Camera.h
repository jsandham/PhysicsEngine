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
		Guid mComponentId;
		Guid mEntityId;
		Guid mTargetTextureId;
		CameraMode mMode;
		glm::vec3 mPosition;
		glm::vec3 mFront;
		glm::vec3 mUp;
		glm::vec4 mBackgroundColor;
		int mX;
		int mY;
		int mWidth;
		int mHeight;
		float mFov;
		float mNearPlane;
		float mFarPlane;
	};
#pragma pack(pop)

	// plane defined by n.x*x + n.y*y + n.z*z + d = 0, where d = -dot(n, x0)
	struct Plane
	{
		glm::vec3 mN;
		glm::vec3 mX0;

		float distance(glm::vec3 point) const;
	};

	struct Viewport
	{
		int mX;
		int mY;
		int mWidth;
		int mHeight;

		float getAspectRatio() const;
	};

	struct Frustum
	{
		Plane mPlanes[6];

		float mFov;
		float mNearPlane;
		float mFarPlane;

		int checkPoint(glm::vec3 point) const;
		int checkSphere(glm::vec3 centre, float radius) const;
		int checkAABB(glm::vec3 min, glm::vec3 max) const;
	};

	class Camera : public Component
	{
		public:
			Frustum mFrustum;
			Viewport mViewport;
			Guid mTargetTextureId;
			
			GLuint mMainFBO;
			GLuint mColorTex;
			GLuint mDepthTex;

			GLuint mGeometryFBO;
			GLuint mPositionTex;
			GLuint mNormalTex;

			GLuint mSsaoFBO;
			GLuint mSsaoColorTex;
		
			GLuint mSsaoNoiseTex;

			std::vector<glm::vec3> mSsaoSamples;

			enum {
				TOP = 0,
				BOTTOM,
				LEFT,
				RIGHT,
				NEAR,
				FAR
			};

			CameraMode mMode;

			glm::vec3 mPosition;
			glm::vec3 mFront;
			glm::vec3 mUp;
			glm::vec3 mRight;
			glm::vec4 mBackgroundColor;

			bool mIsCreated;
			bool mUseSSAO;

		public:
			Camera();
			Camera(std::vector<char> data);
			~Camera();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
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
	struct IsCamera { static const bool value; };

	template <typename T>
	const bool IsCamera<T>::value = false;

	template<>
	const bool IsCamera<Camera>::value = true;
	template<>
	const bool IsComponent<Camera>::value = true;
}

#endif