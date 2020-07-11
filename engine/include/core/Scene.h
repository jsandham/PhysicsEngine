#ifndef __SCENE_H__
#define __SCENE_H__

#include "Guid.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct SceneHeader
	{
		uint64_t mSignature;
		uint32_t mSize;
		uint32_t mEntityCount;
		uint32_t mComponentCount;
		uint32_t mSystemCount;
		uint8_t mMajor;
		uint8_t mMinor;
		Guid mSceneId;
	};
#pragma pack(pop)
}

#endif