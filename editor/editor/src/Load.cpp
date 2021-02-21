#include <iostream>
#include <unordered_map>
#include <vector>

#include <core/Load.h>
#include <core/PoolAllocator.h>

#include "../include/EditorCameraSystem.h"

using namespace PhysicsEngine;

//void PhysicsEngine::addAssetIdToIndexMap(std::unordered_map<Guid, int> *idToGlobalIndex,
//                                         std::unordered_map<Guid, int> *idToType, const Guid &id, int type, int index)
//{
//}
//
//void PhysicsEngine::addComponentIdToIndexMap(std::unordered_map<Guid, int> *idToGlobalIndex,
//                                             std::unordered_map<Guid, int> *idToType, const Guid &id, int type,
//                                             int index)
//{
//}
//
//void PhysicsEngine::addSystemIdToIndexMap(std::unordered_map<Guid, int> *idToGlobalIndex,
//                                          std::unordered_map<Guid, int> *idToType, const Guid &id, int type, int index)
//{
//    if (type == ComponentType<EditorCameraSystem>::type)
//    {
//        (*idToGlobalIndex)[id] = index;
//        (*idToType)[id] = type;
//    }
//}
//
//void PhysicsEngine::removeAssetIdFromIndexMap(std::unordered_map<Guid, int> *idToGlobalIndex,
//                                              std::unordered_map<Guid, int> *idToType, const Guid &id, int type)
//{
//}
//
//void PhysicsEngine::removeComponentIdFromIndexMap(std::unordered_map<Guid, int> *idToGlobalIndex,
//                                                  std::unordered_map<Guid, int> *idToType, const Guid &id, int type)
//{
//}
//
//void PhysicsEngine::removeSystemIdFromIndexMap(std::unordered_map<Guid, int> *idToGlobalIndex,
//                                               std::unordered_map<Guid, int> *idToType, const Guid &id, int type)
//{
//}
//
//Asset *PhysicsEngine::getAsset(std::unordered_map<int, Allocator *> *allocatorMap, int type, int index)
//{
//    return NULL;
//}
//
//Component *PhysicsEngine::getComponent(std::unordered_map<int, Allocator *> *allocatorMap, int type, int index)
//{
//    return NULL;
//}
//
//System *PhysicsEngine::getSystem(std::unordered_map<int, Allocator *> *allocatorMap, int type, int index)
//{
//    /*if (type == SystemType<EditorCameraSystem>::type) {
//        return allocatorMap->get(index);
//    }*/
//
//    return NULL;
//}

Asset *PhysicsEngine::loadAsset(WorldAllocators& allocators, WorldIdState& state, std::istream& in, const Guid& id, int type)
{
    return NULL;
}

Component *PhysicsEngine::loadComponent(WorldAllocators& allocators, WorldIdState& state, std::istream& in, const Guid& id, int type)
{
    return NULL;
}

System *PhysicsEngine::loadSystem(WorldAllocators& allocators, WorldIdState& state, std::istream& in, const Guid& id, int type)
{
    return NULL;
}

Component *PhysicsEngine::destroyComponent(WorldAllocators& allocators, WorldIdState& state, const Guid& id, int type, int index)
{
    return NULL;
}