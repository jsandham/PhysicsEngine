#ifndef WRITEINTERNALTOJSON_H__
#define WRITEINTERNALTOJSON_H__

#include <string>

#include "Guid.h"
#include "World.h"

#include "../json/json.hpp"

namespace PhysicsEngine
{
void writeInternalAssetToJson(json::JSON &obj, World *world, Guid assetId, int type);
void writeInternalEntityToJson(json::JSON &obj, World *world, Guid entityId);
void writeInternalComponentToJson(json::JSON &obj, World *world, Guid entityId, Guid componentId, int type);
void writeInternalSystemToJson(json::JSON &obj, World *world, Guid systemId, int type, int order);

template <int T> void writeInternalAssetToJson(json::JSON &obj, World *world, Guid assetId)
{
}

template <> inline void writeInternalAssetToJson<1>(json::JSON &obj, World *world, Guid assetId)
{
}

} // namespace PhysicsEngine

#endif