#ifndef __LIGHT_H__
#define __LIGHT_H__

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtc/matrix_transform.hpp"

#include "Component.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct LightHeader
	{
		Guid mComponentId;
		Guid mEntityId;
		glm::vec3 mAmbient;
		glm::vec3 mDiffuse;
		glm::vec3 mSpecular;
		float mConstant;
		float mLinear;
		float mQuadratic;
		float mCutOff;
		float mOuterCutOff;
		uint8_t mLightType;
		uint8_t mShadowType;
	};
#pragma pack(pop)
	
	enum class LightType
	{
		Directional,
		Spot,
		Point,
		None
	};

	enum class ShadowType
	{
		Hard,
		Soft,
		None
	};

	class Light : public Component
	{
		public:
			glm::vec3 mAmbient;
			glm::vec3 mDiffuse;
			glm::vec3 mSpecular;
			float mConstant;
			float mLinear;
			float mQuadratic;
			float mCutOff;
			float mOuterCutOff;
			LightType mLightType;
			ShadowType mShadowType;

		public:
			Light();
			Light(const std::vector<char>& data);
			~Light();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid componentId, Guid entityId) const;
			void deserialize(const std::vector<char>& data);

			glm::mat4 getProjMatrix() const;
	};

	template <typename T>
	struct IsLight { static constexpr bool value = false; };

	template <>
	struct ComponentType<Light> { static constexpr int type = PhysicsEngine::LIGHT_TYPE;};
	template <>
	struct IsLight<Light> { static constexpr bool value = true; };
	template <>
	struct IsComponent<Light> { static constexpr bool value = true; };
	template <>
	struct IsComponentInternal<Light> { static constexpr bool value = true; };
}

#endif