//#include <fstream>
//#include <iostream>
//#include <sstream>
//
//#include "../../include/core/Load.h"
//#include "../../include/core/LoadInternal.h"
//#include "../../include/core/Log.h"
//#include "../../include/core/Serialize.h"
//#include "../../include/core/WorldSerialization.h"
//
//using namespace PhysicsEngine;
//
//const uint64_t PhysicsEngine::ASSET_FILE_SIGNATURE = 0x9a9e9b4153534554;
//const uint64_t PhysicsEngine::SCENE_FILE_SIGNATURE = 0x9a9e9b5343454e45;
//
//void PhysicsEngine::loadAssetIntoWorld(const std::string &filepath, WorldAllocators &allocators, WorldIsState &idState,
//                                       std::unordered_map<Guid, std::string> &assetIdToFilepath)
//{
//    std::ifstream file;
//    file.open(filepath, std::ios::binary);
//
//    if (!file.is_open())
//    {
//        std::string errorMessage = "Failed to open asset bundle " + filepath + "\n";
//        Log::error(&errorMessage[0]);
//        return;
//    }
//
//    AssetFileHeader header;
//    file.read(reinterpret_cast<char *>(&header), sizeof(AssetFileHeader));
//
//    assert(header.mSignature == ASSET_FILE_SIGNATURE && "Trying to load an invalid binary asset file\n");
//
//    //std::vector<char> data(header.mSize);
//    //file.read(reinterpret_cast<char *>(&data[0]), data.size() * sizeof(char));
//    //file.close();
//
//    Asset *asset = NULL;
//
//    std::unordered_map<Guid, int>::iterator it = idState.mIdToGlobalIndex->find(header.mAssetId);
//    if (it != idState.mIdToGlobalIndex->end())
//    {
//        if (Asset::isInternal(header.mType))
//        {
//            asset = PhysicsEngine::getInternalAsset(allocators.mMeshAllocator, allocators.mMaterialAllocator,
//                                                    allocators.mShaderAllocator, allocators.mTexture2DAllocator,
//                                                    allocators.mTexture3DAllocator, allocators.mCubemapAllocator,
//                                                    allocators.mFontAllocator, header.mType, it->second);
//        }
//        else
//        {
//            asset = PhysicsEngine::getAsset(allocators.mAssetAllocatorMap, header.mType, it->second);
//        }
//
//        assert(asset != NULL && "Could not find asset\n");
//
//        //asset->deserialize(data);
//        asset->deserialize(file);
//    }
//    else
//    {
//        int index = -1;
//        if (Asset::isInternal(header.mType))
//        {
//            asset = PhysicsEngine::loadInternalAsset(allocators.mMeshAllocator, allocators.mMaterialAllocator,
//                                                     allocators.mShaderAllocator, allocators.mTexture2DAllocator,
//                                                     allocators.mTexture3DAllocator, allocators.mCubemapAllocator,
//                                                     allocators.mFontAllocator, file, header.mType, &index);
//        }
//        else
//        {
//            asset = PhysicsEngine::loadAsset(allocators.mAssetAllocatorMap, file, header.mType, &index);
//        }
//
//        assert(asset != NULL && "Returned a NULL asset after loading\n");
//        assert(index >= 0 && "Returned a negative index for asset after loading\n");
//
//        if (Asset::isInternal(header.mType))
//        {
//            PhysicsEngine::addInternalAssetIdToIndexMap(
//                idState.mMeshIdToGlobalIndex, idState.mMaterialIdToGlobalIndex, idState.mShaderIdToGlobalIndex,
//                idState.mTexture2DIdToGlobalIndex, idState.mTexture3DIdToGlobalIndex, idState.mCubemapIdToGlobalIndex,
//                idState.mFontIdToGlobalIndex, idState.mIdToGlobalIndex, idState.mIdToType, asset->getId(), header.mType,
//                index);
//        }
//        else
//        {
//        }
//
//        assetIdToFilepath[asset->getId()] = filepath;
//    }
//
//    file.close();
//}
//
//void PhysicsEngine::loadSceneIntoWorld(const std::string &filepath, World& world)
//{
//    std::ifstream file;
//    file.open(filepath, std::ios::binary);
//
//    if (!file.is_open())
//    {
//        std::string errorMessage = "Failed to open scene file " + filepath + "\n";
//        Log::error(&errorMessage[0]);
//        return;
//    }
//
//    SceneHeader sceneHeader;
//    PhysicsEngine::read<SceneHeader>(file, sceneHeader);
//
//    assert(sceneHeader.mSignature == SCENE_FILE_SIGNATURE && "Trying to load an invalid binary scene file\n");
//
//    size_t entityCount = sceneHeader.mEntityCount;
//    size_t componentCount = sceneHeader.mComponentCount;
//    size_t systemCount = sceneHeader.mSystemCount;
//
//    size_t count = entityCount + componentCount + systemCount;
//
//    std::vector<ObjectHeader> headers(count);
//
//    for (size_t i = 0; i < headers.size(); i++) {
//        PhysicsEngine::read<ObjectHeader>(file, headers[i]);
//    }
//
//    for (size_t i = 0; i < entityCount; i++) {
//        PhysicsEngine::loadEntity(file);
//    }
//
//    for (size_t i = 0; i < componentCount; i++) {
//        PhysicsEngine::loadComponent(file);
//    }
//
//    for (size_t i = 0; i < systemCount; i++) {
//        PhysicsEngine::loadSystem(file);
//    }
//
//
//        
//
//    for (size_t i = 0; i < componentHeaders.size(); i++) {
//        Component* component = world.getComponent(componentHeaders[i].mType, componentHeaders[i].mId);
//
//        if (component != nullptr) {
//            component->deserialize(file);
//        }
//        else {
//            world.addComponent(componentHeaders[i].mType, file);
//        }
//    }
//
//    for (size_t i = 0; i < systemHeaders.size(); i++) {
//        System* system = world.getSystem(systemHeaders[i].mType, systemHeaders[i].mSystemId);
//
//        if (system != nullptr) {
//            system->deserialize(file);
//        }
//        else {
//            world.addSystem(systemHeaders[i].mType, file);
//        }
//    }
//
//
//
//
//
//
//
//
//
//    std::vector<char> data(sceneHeader.mSize);
//    file.read(reinterpret_cast<char *>(&data[0]), data.size() * sizeof(char));
//    file.close();
//
//    //size_t start = 0;
//    //size_t end = 0;
//
//    //std::vector<ComponentInfoHeader> componentInfoHeaders(sceneHeader.mComponentCount);
//    //std::vector<SystemInfoHeader> systemInfoHeaders(sceneHeader.mSystemCount);
//
//    //// load all component info headers
//    //for (int32_t i = 0; i < sceneHeader.mComponentCount; i++)
//    //{
//    //    start = end;
//    //    end += sizeof(ComponentInfoHeader);
//
//    //    std::vector<char> componentInfoData(&data[start], &data[end]);
//
//    //    componentInfoHeaders[i] = *reinterpret_cast<ComponentInfoHeader *>(componentInfoData.data());
//    //}
//
//    //// load all system info headers
//    //for (int32_t i = 0; i < sceneHeader.mSystemCount; i++)
//    //{
//    //    start = end;
//    //    end += sizeof(SystemInfoHeader);
//
//    //    std::vector<char> systemInfoData(&data[start], &data[end]);
//
//    //    systemInfoHeaders[i] = *reinterpret_cast<SystemInfoHeader *>(systemInfoData.data());
//    //}
//
//    //// load all entities
//    //for (int32_t i = 0; i < sceneHeader.mEntityCount; i++)
//    //{
//    //    start = end;
//    //    end += sizeof(EntityHeader);
//
//    //    std::vector<char> entityData(&data[start], &data[end]);
//
//    //    EntityHeader *entityHeader =
//    //        reinterpret_cast<EntityHeader *>(entityData.data()); // use endian agnostic function to read header
//
//    //    std::unordered_map<Guid, int>::iterator it = idState.mIdToGlobalIndex->find(entityHeader->mEntityId);
//    //    if (it != idState.mIdToGlobalIndex->end())
//    //    {
//    //        Entity *entity = PhysicsEngine::getInternalEntity(allocators.mEntityAllocator, it->second);
//
//    //        assert(entity != NULL && "Could not find entity\n");
//
//    //        entity->deserialize(entityData);
//    //    }
//    //    else
//    //    {
//    //        int index = -1;
//    //        Entity *entity = PhysicsEngine::loadInternalEntity(allocators.mEntityAllocator, entityData, &index);
//
//    //        assert(entity != NULL && "Returned a NULL entity after loading\n");
//    //        assert(index >= 0 && "Returned a negative index for entity after loading\n");
//
//    //        PhysicsEngine::addInternalEntityIdToIndexMap(idState.mEntityIdToGlobalIndex, idState.mIdToGlobalIndex,
//    //                                                     idState.mIdToType, entity->getId(), index);
//
//    //        (*idState.mEntityIdToComponentIds)[entity->getId()] = std::vector<std::pair<Guid, int>>();
//    //        (*idState.mEntityIdsMarkedCreated).push_back(entity->getId());
//    //    }
//    //}
//
//    //// load all components
//    //for (int32_t i = 0; i < sceneHeader.mComponentCount; i++)
//    //{
//    //    std::vector<char> temp(&data[componentInfoHeaders[i].mStartPtr],
//    //                           &data[componentInfoHeaders[i].mStartPtr + componentInfoHeaders[i].mSize]);
//
//    //    std::unordered_map<Guid, int>::iterator it =
//    //        idState.mIdToGlobalIndex->find(componentInfoHeaders[i].mComponentId);
//    //    if (it != idState.mIdToGlobalIndex->end())
//    //    {
//    //        Component *component = NULL;
//    //        if (Component::isInternal(componentInfoHeaders[i].mType))
//    //        {
//    //            component = PhysicsEngine::getInternalComponent(
//    //                allocators.mTransformAllocator, allocators.mMeshRendererAllocator,
//    //                allocators.mLineRendererAllocator, allocators.mRigidbodyAllocator, allocators.mCameraAllocator,
//    //                allocators.mLightAllocator, allocators.mSphereColliderAllocator, allocators.mBoxColliderAllocator,
//    //                allocators.mCapsuleColliderAllocator, allocators.mMeshColliderAllocator,
//    //                componentInfoHeaders[i].mType, it->second);
//    //        }
//    //        else
//    //        {
//    //            component = PhysicsEngine::getComponent(allocators.mComponentAllocatorMap,
//    //                                                    componentInfoHeaders[i].mType, it->second);
//    //        }
//
//    //        assert(component != NULL && "Could not find component\n");
//
//    //        component->deserialize(temp);
//    //    }
//    //    else
//    //    {
//    //        Component *component = NULL;
//    //        int index = -1;
//    //        if (Component::isInternal(componentInfoHeaders[i].mType))
//    //        {
//    //            component = PhysicsEngine::loadInternalComponent(
//    //                allocators.mTransformAllocator, allocators.mMeshRendererAllocator,
//    //                allocators.mLineRendererAllocator, allocators.mRigidbodyAllocator, allocators.mCameraAllocator,
//    //                allocators.mLightAllocator, allocators.mSphereColliderAllocator, allocators.mBoxColliderAllocator,
//    //                allocators.mCapsuleColliderAllocator, allocators.mMeshColliderAllocator, temp,
//    //                componentInfoHeaders[i].mType, &index);
//    //        }
//    //        else
//    //        {
//    //            component = PhysicsEngine::loadComponent(allocators.mComponentAllocatorMap, temp,
//    //                                                     componentInfoHeaders[i].mType, &index);
//    //        }
//
//    //        assert(component != NULL && "Returned a NULL component after loading\n");
//    //        assert(index >= 0 && "Returned a negative index for component after loading\n");
//
//    //        if (Component::isInternal(componentInfoHeaders[i].mType))
//    //        {
//    //            PhysicsEngine::addInternalComponentIdToIndexMap(
//    //                idState.mTransformIdToGlobalIndex, idState.mMeshRendererIdToGlobalIndex,
//    //                idState.mLineRendererIdToGlobalIndex, idState.mRigidbodyIdToGlobalIndex,
//    //                idState.mCameraIdToGlobalIndex, idState.mLightIdToGlobalIndex,
//    //                idState.mSphereColliderIdToGlobalIndex, idState.mBoxColliderIdToGlobalIndex,
//    //                idState.mCapsuleColliderIdToGlobalIndex, idState.mMeshColliderIdToGlobalIndex,
//    //                idState.mIdToGlobalIndex, idState.mIdToType, component->getId(), componentInfoHeaders[i].mType,
//    //                index);
//    //        }
//    //        else
//    //        {
//    //            PhysicsEngine::addComponentIdToIndexMap(idState.mIdToGlobalIndex, idState.mIdToType, component->getId(),
//    //                                                    componentInfoHeaders[i].mType, index);
//    //        }
//
//    //        (*idState.mEntityIdToComponentIds)[component->getEntityId()].push_back(
//    //            std::make_pair(component->getId(), componentInfoHeaders[i].mType));
//    //        (*idState.mComponentIdsMarkedCreated)
//    //            .push_back(make_triple(component->getEntityId(), component->getId(), componentInfoHeaders[i].mType));
//    //    }
//    //}
//
//    //// load all systems
//    //for (int32_t i = 0; i < sceneHeader.mSystemCount; i++)
//    //{
//    //    std::vector<char> temp(&data[systemInfoHeaders[i].mStartPtr],
//    //                           &data[systemInfoHeaders[i].mStartPtr + systemInfoHeaders[i].mSize]);
//
//    //    std::unordered_map<Guid, int>::iterator it = idState.mIdToGlobalIndex->find(systemInfoHeaders[i].mSystemId);
//    //    if (it != idState.mIdToGlobalIndex->end())
//    //    {
//    //        System *system = NULL;
//    //        if (systemInfoHeaders[i].mType <= PhysicsEngine::MAX_INTERNAL_SYSTEM)
//    //        {
//    //            system = PhysicsEngine::getInternalSystem(
//    //                allocators.mRenderSystemAllocator, allocators.mPhysicsSystemAllocator,
//    //                allocators.mCleanupSystemAllocator, allocators.mDebugSystemAllocator,
//    //                allocators.mGizmoSystemAllocator, systemInfoHeaders[i].mType, it->second);
//    //        }
//    //        else
//    //        {
//    //            system =
//    //                PhysicsEngine::getSystem(allocators.mSystemAllocatorMap, systemInfoHeaders[i].mType, it->second);
//    //        }
//
//    //        assert(system != NULL && "Could not find system\n");
//
//    //        system->deserialize(temp);
//    //    }
//    //    else
//    //    {
//    //        System *system = NULL;
//    //        int index = -1;
//    //        if (systemInfoHeaders[i].mType <= PhysicsEngine::MAX_INTERNAL_SYSTEM)
//    //        {
//    //            system = PhysicsEngine::loadInternalSystem(
//    //                allocators.mRenderSystemAllocator, allocators.mPhysicsSystemAllocator,
//    //                allocators.mCleanupSystemAllocator, allocators.mDebugSystemAllocator,
//    //                allocators.mGizmoSystemAllocator, temp, systemInfoHeaders[i].mType, &index);
//    //        }
//    //        else
//    //        {
//    //            system =
//    //                PhysicsEngine::loadSystem(allocators.mSystemAllocatorMap, temp, systemInfoHeaders[i].mType, &index);
//    //        }
//
//    //        assert(system != NULL && "Returned a NULL system after loading\n");
//    //        assert(index >= 0 && "Returned a negative index for system after loading\n");
//
//    //        if (systemInfoHeaders[i].mType <= PhysicsEngine::MAX_INTERNAL_SYSTEM)
//    //        {
//    //            PhysicsEngine::addInternalSystemIdToIndexMap(
//    //                idState.mRenderSystemIdToGlobalIndex, idState.mPhysicsSystemIdToGlobalIndex,
//    //                idState.mCleanupSystemIdToGlobalIndex, idState.mDebugSystemIdToGlobalIndex,
//    //                idState.mGizmoSystemIdToGlobalIndex, idState.mIdToGlobalIndex, idState.mIdToType, system->getId(),
//    //                systemInfoHeaders[i].mType, index);
//    //        }
//    //        else
//    //        {
//    //            PhysicsEngine::addSystemIdToIndexMap(idState.mIdToGlobalIndex, idState.mIdToType, system->getId(),
//    //                                                 systemInfoHeaders[i].mType, index);
//    //        }
//    //    }
//    //}
//}
//
//
//
//
//
//void PhysicsEngine::loadEntity(World& world, std::ifstream& in)
//{
//
//}
//
//void PhysicsEngine::loadComponent(World& world, std::ifstream& in);
//void PhysicsEngine::loadSystem(World& world, std::ifstream& in);