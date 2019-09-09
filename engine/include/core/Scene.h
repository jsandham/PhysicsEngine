#ifndef __SCENE_H__
#define __SCENE_H__

#include <string>

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct SceneHeader
	{
		unsigned short fileType;
		unsigned int fileSize;
	};
#pragma pack(pop)
}

#endif