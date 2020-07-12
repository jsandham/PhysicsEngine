#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <vector>
#include <unordered_map>

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
#include "../core/Color.h"

#include "../graphics/GraphicsTargets.h"
#include "../graphics/GraphicsQuery.h"

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

	enum class RenderPath
	{
		Forward,
		Deferred
	};

#pragma pack(push, 1)
	struct CameraHeader
	{
		Guid mComponentId;
		Guid mEntityId;
		Guid mTargetTextureId;
		glm::vec4 mBackgroundColor;
		int32_t mX;
		int32_t mY;
		int32_t mWidth;
		int32_t mHeight;
		float mFov;
		float mAspectRatio;
		float mNearPlane;
		float mFarPlane;
		uint8_t mRenderPath;
		uint8_t mMode;
		uint8_t mSSAO;
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

			RenderPath mRenderPath;
			CameraMode mMode;
			CameraSSAO mSSAO;

			Color mBackgroundColor;

			GraphicsQuery mQuery;

		private:
			GraphicsTargets mTargets;

			glm::vec3 mSsaoSamples[64];
			glm::mat4 viewMatrix;

			bool mIsCreated;

			std::unordered_map<int, Guid> mColoringMap;

		public:
			Camera();
			Camera(const std::vector<char>& data);
			~Camera();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(const std::vector<char>& data);

			void create();
			void destroy();
			void computeViewMatrix(glm::vec3 position, glm::vec3 forward, glm::vec3 up);
			void assignColoring(int color, Guid meshRendererId);
			void clearColoring();

			void beginQuery();
			void endQuery();

			bool isCreated() const;
			glm::mat4 getViewMatrix() const;
			glm::mat4 getProjMatrix() const;
			glm::vec3 getSSAOSample(int sample) const;
			Guid getMeshRendererIdAtScreenPos(int x, int y) const;

			GLuint getNativeGraphicsMainFBO() const;
			GLuint getNativeGraphicsColorPickingFBO() const;
			GLuint getNativeGraphicsGeometryFBO() const;
			GLuint getNativeGraphicsSSAOFBO() const;

			GLuint getNativeGraphicsColorTex() const;
			GLuint getNativeGraphicsDepthTex() const;
			GLuint getNativeGraphicsColorPickingTex() const;
			GLuint getNativeGraphicsPositionTex() const;
			GLuint getNativeGraphicsNormalTex() const;
			GLuint getNativeGraphicsAlbedoSpecTex() const;
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
	template<>
	const bool IsComponentInternal<Camera>::value = true;
}

#endif