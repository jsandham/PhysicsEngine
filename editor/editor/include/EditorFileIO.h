#ifndef __EDITOR_WORLD_UTIL_H__
#define __EDITOR_WORLD_UTIL_H__

#include <set>
#include <string>

#include "core/Guid.h"
#include "core/World.h"

namespace PhysicsEditor
{
bool writeAssetToBinary(std::string filePath, std::string extension, PhysicsEngine::Guid id, std::string outFilePath);
bool writeSceneToBinary(std::string filePath, PhysicsEngine::Guid id, std::string outFilePath);
bool writeAssetToJson(PhysicsEngine::World *world, std::string outfilePath, PhysicsEngine::Guid assetId, int type);
bool writeSceneToJson(PhysicsEngine::World *world, std::string outfilePath,
                      std::set<PhysicsEngine::Guid> editorOnlyEntityIds);

bool createMetaFile(std::string metaFilePath);
PhysicsEngine::Guid findGuidFromMetaFilePath(std::string metaFilePath);
} // namespace PhysicsEditor

#endif
