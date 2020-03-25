#include "../../include/components/Light.h"

using namespace PhysicsEngine;

Light::Light()
{
	mComponentId = Guid::INVALID;
	mEntityId = Guid::INVALID;

	mPosition = glm::vec3(0.0f, 1.0f, 0.0f);
	mDirection = glm::vec3(1.0f, 2.0f, 0.0f);
	mAmbient = glm::vec3(0.4f, 0.4f, 0.4f);
	mDiffuse = glm::vec3(1.0f, 1.0f, 1.0f);
	mSpecular = glm::vec3(1.0f, 1.0f, 1.0f);
	mConstant = 1.0f;
	mLinear = 0.1f;
	mQuadratic = 0.032f;
	mCutOff = glm::cos(glm::radians(12.5f));
	mOuterCutOff = glm::cos(glm::radians(15.0f));
	mLightType = LightType::Directional;
	mShadowType = ShadowType::Hard;
}

Light::Light(std::vector<char> data)
{
	deserialize(data);
}

Light::~Light()
{

}

std::vector<char> Light::serialize() const
{
	return serialize(mComponentId, mEntityId);
}

std::vector<char> Light::serialize(Guid componentId, Guid entityId) const
{
	LightHeader header;
	header.mComponentId = componentId;
	header.mEntityId = entityId;
	header.mPosition = mPosition;
	header.mDirection = mDirection;
	header.mAmbient = mAmbient;
	header.mDiffuse = mDiffuse;
	header.mSpecular = mSpecular;
	header.mConstant = mConstant;
	header.mLinear = mLinear;
	header.mQuadratic = mQuadratic;
	header.mCutOff = mCutOff;
	header.mOuterCutOff = mOuterCutOff;
	header.mLightType = static_cast<int>(mLightType);
	header.mShadowType = static_cast<int>(mShadowType);

	std::vector<char> data(sizeof(LightHeader));

	memcpy(&data[0], &header, sizeof(LightHeader));

	return data;
}

void Light::deserialize(std::vector<char> data)
{
	LightHeader* header = reinterpret_cast<LightHeader*>(&data[0]);

	mComponentId = header->mComponentId;
	mEntityId = header->mEntityId;

	mPosition = header->mPosition;
	mDirection = header->mDirection;
	mAmbient = header->mAmbient;
	mDiffuse = header->mDiffuse;
	mSpecular = header->mSpecular;

	mConstant = header->mConstant;
	mLinear = header->mLinear;
	mQuadratic = header->mQuadratic;
	mCutOff = glm::cos(glm::radians(header->mCutOff));
	mOuterCutOff = glm::cos(glm::radians(header->mOuterCutOff));

	mLightType = static_cast<LightType>(header->mLightType);
	mShadowType = static_cast<ShadowType>(header->mShadowType);
}

glm::mat4 Light::getProjMatrix() const
{
	return glm::perspective(2.0f * glm::radians(mOuterCutOff), 1.0f * 1024 / 1024, 0.1f, 12.0f);
}