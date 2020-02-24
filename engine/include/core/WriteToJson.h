#ifndef __WRITETOJSON_H__
#define __WRITETOJSON_H__

#include <string>

#include "World.h"
#include "Guid.h"

#include "../json/json.hpp"

namespace PhysicsEngine
{
	void writeComponentToJson(json::JSON& obj, World* world, Guid entityId, Guid componentId, int type);
	void writeSystemToJson(json::JSON& obj, World* world, Guid systemId, int type, int order);
}

#endif