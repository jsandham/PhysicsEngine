#ifndef __LOAD_H__
#define __LOAD_H__

#include <unordered_map>
#include <vector>

#include "../components/Component.h"
#include "../systems/System.h"
#include "Allocator.h"
#include "Asset.h"

namespace PhysicsEngine
{
// add external asset defined by the user to global index map (assets with type 10 or greater)
void addAssetIdToIndexMap(std::unordered_map<Guid, int> *idToGlobalIndex, std::unordered_map<Guid, int> *idToTypeMap,
                          const Guid &id, int type, int index);

// add external component defined by the user to global index map (assets with type 10 or greater)
void addComponentIdToIndexMap(std::unordered_map<Guid, int> *idToGlobalIndex,
                              std::unordered_map<Guid, int> *idToTypeMap, const Guid &id, int type, int index);

// add external system defined by the user to global index map (assets with type 10 or greater)
void addSystemIdToIndexMap(std::unordered_map<Guid, int> *idToGlobalIndex, std::unordered_map<Guid, int> *idToTypeMap,
                           const Guid &id, int type, int index);

// remove external asset defined by the user to global index map (assets with type 10 or greater)
void removeAssetIdFromIndexMap(std::unordered_map<Guid, int> *idToGlobalIndex,
                               std::unordered_map<Guid, int> *idToTypeMap, const Guid &id, int type);

// remove external component defined by the user to global index map (assets with type 10 or greater)
void removeComponentIdFromIndexMap(std::unordered_map<Guid, int> *idToGlobalIndex,
                                   std::unordered_map<Guid, int> *idToTypeMap, const Guid &id, int type);

// remove external system defined by the user to global index map (assets with type 10 or greater)
void removeSystemIdFromIndexMap(std::unordered_map<Guid, int> *idToGlobalIndex,
                                std::unordered_map<Guid, int> *idToTypeMap, const Guid &id, int type);

// get external asset defined by the user (assets with type 10 or greater)
Asset *getAsset(std::unordered_map<int, Allocator *> *allocatorMap, int type, int index);

// get external components defined by the user (components with type 20 or greater)
Component *getComponent(std::unordered_map<int, Allocator *> *allocatorMap, int type, int index);

// get external systems defined by the user (systems with type 10 or greater)
System *getSystem(std::unordered_map<int, Allocator *> *allocatorMap, int type, int index);

// load external asset defined by the user (assets with type 10 or greater)
Asset *loadAsset(std::unordered_map<int, Allocator *> *allocatorMap, const std::vector<char> &data, int type,
                 int *index);

// load external components defined by the user (components with type 20 or greater)
Component *loadComponent(std::unordered_map<int, Allocator *> *allocatorMap, const std::vector<char> &data, int type,
                         int *index);

// load external systems defined by the user (systems with type 10 or greater)
System *loadSystem(std::unordered_map<int, Allocator *> *allocatorMap, const std::vector<char> &data, int type,
                   int *index);

// destroy external components defined by the user
Component *destroyComponent(std::unordered_map<int, Allocator *> *allocatorMap, int type, int index);
} // namespace PhysicsEngine

#endif