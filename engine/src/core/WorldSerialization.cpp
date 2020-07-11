#include <iostream>
#include <fstream>
#include <sstream>

#include "../../include/core/WorldSerialization.h"
#include "../../include/core/Log.h"
#include "../../include/core/LoadInternal.h"
#include "../../include/core/Load.h"
#include "../../include/core/Scene.h"

using namespace PhysicsEngine;

void PhysicsEngine::loadAssetIntoWorld(const std::string& filepath,
										PoolAllocator<Mesh>& meshAllocator,
										PoolAllocator<Material>& materialAllocator,
										PoolAllocator<Shader>& shaderAllocator,
										PoolAllocator<Texture2D>& texture2DAllocator,
										PoolAllocator<Texture3D>& texture3DAllocator,
										PoolAllocator<Cubemap>& cubemapAllocator,
										PoolAllocator<Font>& fontAllocator,
										std::unordered_map<int, Allocator*>& assetAllocatorMap,
										std::unordered_map<Guid, int>& idToGlobalIndex,
										std::unordered_map<Guid, int>& idToType,
										std::unordered_map<Guid, std::string>& assetIdToFilepath)
{
	std::ifstream file;
	file.open(filepath, std::ios::binary);

	if (!file.is_open()) {
		std::string errorMessage = "Failed to open asset bundle " + filepath + "\n";
		Log::error(&errorMessage[0]);
		return;
	}

	AssetHeader header;
	file.read(reinterpret_cast<char*>(&header), sizeof(AssetHeader));

	assert(header.mSignature == 0x9a9e9b4153534554 && "Trying to load an invalid binary asset file\n");

	std::vector<char> data(header.mSize);
	file.read(reinterpret_cast<char*>(&data[0]), data.size() * sizeof(char));
	file.close();
	
	Asset* asset = NULL;

	std::unordered_map<Guid, int>::iterator it = idToGlobalIndex.find(header.mAssetId);
	if (it != idToGlobalIndex.end()) {
		if (header.mType < 20) {
			asset = PhysicsEngine::getInternalAsset(&meshAllocator,
				&materialAllocator,
				&shaderAllocator,
				&texture2DAllocator,
				&texture3DAllocator,
				&cubemapAllocator,
				&fontAllocator,
				header.mType,
				it->second);
		}
		else {
			asset = PhysicsEngine::getAsset(&assetAllocatorMap, header.mType, it->second);
		}
		
		assert(asset != NULL && "Could not find asset\n");

		asset->deserialize(data);
	}
	else
	{
		int index = -1;
		if (header.mType < 20) {
			asset = PhysicsEngine::loadInternalAsset(&meshAllocator,
				&materialAllocator,
				&shaderAllocator,
				&texture2DAllocator,
				&texture3DAllocator,
				&cubemapAllocator,
				&fontAllocator,
				data,
				header.mType,
				&index);
		}
		else {
			asset = PhysicsEngine::loadAsset(&assetAllocatorMap, data, header.mType, &index);
		}

		assert(asset != NULL && "Returned a NULL asset after loading\n");
		assert(index >= 0 && "Returned a negative index for asset after loading\n");

		idToGlobalIndex[asset->getId()] = index;
		idToType[asset->getId()] = header.mType;
		assetIdToFilepath[asset->getId()] = filepath;
	}
}

void PhysicsEngine::loadSceneIntoWorld(const std::string& filepath,
					PoolAllocator<Entity>& entityAllocator,
					PoolAllocator<Transform>& transformAllocator,
					PoolAllocator<MeshRenderer>& meshRendererAllocator,
					PoolAllocator<LineRenderer>& lineRendererAllocator,
					PoolAllocator<Rigidbody>& rigidbodyAllocator,
					PoolAllocator<Camera>& cameraAllocator,
					PoolAllocator<Light>& lightAllocator,
					PoolAllocator<SphereCollider>& sphereColliderAllocator,
					PoolAllocator<BoxCollider>& boxColliderAllocator,
					PoolAllocator<CapsuleCollider>& capsuleColliderAllocator,
					PoolAllocator<MeshCollider>& meshColliderAllocator,
					PoolAllocator<RenderSystem>& renderSystemAllocator,
					PoolAllocator<PhysicsSystem>& physicsSystemAllocator,
					PoolAllocator<CleanUpSystem>& cleanupSystemAllocator,
					PoolAllocator<DebugSystem>& debugSystemAllocator,
					std::unordered_map<int, Allocator*>& componentAllocatorMap,
					std::unordered_map<int, Allocator*>& systemAllocatorMap,
					std::unordered_map<Guid, int>& idToGlobalIndex,
					std::unordered_map<Guid, int>& idToType,
					std::unordered_map<Guid, std::string>& sceneIdToFilepath)
{
	std::ifstream file;
	file.open(filepath, std::ios::binary);

	if (!file.is_open()) {
		std::string errorMessage = "Failed to open scene file " + filepath + "\n";
		Log::error(&errorMessage[0]);
		return;
	}

	SceneHeader sceneHeader;
	file.read(reinterpret_cast<char*>(&sceneHeader), sizeof(SceneHeader));

	assert(sceneHeader.mSignature == 0x9a9e9b5343454e45 && "Trying to load an invalid binary scene file\n");

	std::vector<char> data(sceneHeader.mSize);
	file.read(reinterpret_cast<char*>(&data[0]), data.size() * sizeof(char));
	file.close();

	size_t start = 0;
	size_t end = 0;

	// load all entities
	for (uint32_t i = 0; i < sceneHeader.mEntityCount; i++) {
		start = end;
		end += sizeof(EntityHeader);

		std::vector<char> entityData(&data[start], &data[end]);

		EntityHeader* entityHeader = reinterpret_cast<EntityHeader*>(&entityData); // use endian agnostic function to read header

		std::unordered_map<Guid, int>::iterator it = idToGlobalIndex.find(entityHeader->mEntityId);
		if (it != idToGlobalIndex.end()) {
			Entity* entity = PhysicsEngine::getInternalEntity(&entityAllocator, it->second);

			assert(entity != NULL && "Could not find entity\n");

			entity->deserialize(entityData);
		}
		else
		{
			int index = -1;
			Entity* entity = PhysicsEngine::loadInternalEntity(&entityAllocator, entityData, &index);

			assert(entity != NULL && "Returned a NULL entity after loading\n");
			assert(index >= 0 && "Returned a negative index for entity after loading\n");

			idToGlobalIndex[entity->getId()] = index;
			idToType[entity->getId()] = entityHeader->mType;
		}
	}

	// load all components
	for (uint32_t i = 0; i < sceneHeader.mComponentCount; i++) {
		start = end;
		end += sizeof(ComponentHeader);

		std::vector<char> componentData(&data[start], &data[end]);

		ComponentHeader* componentHeader = reinterpret_cast<ComponentHeader*>(&componentData);

		std::vector<char> temp(&data[componentHeader->mStartPtr], &data[componentHeader->mStartPtr + componentHeader->mSize]);

		std::unordered_map<Guid, int>::iterator it = idToGlobalIndex.find(componentHeader->mComponentId);
		if (it != idToGlobalIndex.end()) {
			Component* component = NULL;
			if (componentHeader->mType < 20) {
				component = PhysicsEngine::getInternalComponent(&transformAllocator,
															&meshRendererAllocator,
															&lineRendererAllocator,
															&rigidbodyAllocator,
															&cameraAllocator,
															&lightAllocator,
															&sphereColliderAllocator,
															&boxColliderAllocator,
															&capsuleColliderAllocator,
															&meshColliderAllocator,
															componentHeader->mType,
															it->second);
			}
			else {
				component = PhysicsEngine::getComponent(&componentAllocatorMap,
														componentHeader->mType,
														it->second);
			}

			assert(component != NULL && "Could not find component\n");

			component->deserialize(temp);
		}
		else
		{
			Component* component = NULL;
			int index = -1;
			if (componentHeader->mType < 20) {
				component = PhysicsEngine::loadInternalComponent(&transformAllocator,
															&meshRendererAllocator,
															&lineRendererAllocator,
															&rigidbodyAllocator,
															&cameraAllocator,
															&lightAllocator,
															&sphereColliderAllocator,
															&boxColliderAllocator,
															&capsuleColliderAllocator,
															&meshColliderAllocator,
															temp,
															componentHeader->mType,
															&index);
			}
			else
			{
				component = PhysicsEngine::loadComponent(&componentAllocatorMap,
														temp,
														componentHeader->mType,
														&index);
			}

			assert(component != NULL && "Returned a NULL component after loading\n");
			assert(index >= 0 && "Returned a negative index for component after loading\n");

			idToGlobalIndex[component->getId()] = index;
			idToType[component->getId()] = componentHeader->mType;
		}
	}

	// load all systems
	for (uint32_t i = 0; i < sceneHeader.mSystemCount; i++) {
		start = end;
		end += sizeof(SystemHeader);

		std::vector<char> systemData(&data[start], &data[end]);

		SystemHeader* systemHeader = reinterpret_cast<SystemHeader*>(&systemData);

		std::unordered_map<Guid, int>::iterator it = idToGlobalIndex.find(systemHeader->mSystemId);
		if (it != idToGlobalIndex.end()) {
			System* system = NULL;
			if (systemHeader->mType < 20) {
				system = PhysicsEngine::getInternalSystem(&renderSystemAllocator,
														  &physicsSystemAllocator,
														  &cleanupSystemAllocator,
														  &debugSystemAllocator,
														  systemHeader->mType,
														  it->second);
			}
			else {
				system = PhysicsEngine::getSystem(&systemAllocatorMap,
												  systemHeader->mType,
												  it->second);
			}

			assert(system != NULL && "Could not find system\n");

			system->deserialize(systemData);
		}
		else
		{
			System* system = NULL;
			int index = -1;
			if (systemHeader->mType < 20) {
				system = PhysicsEngine::loadInternalSystem(&renderSystemAllocator,
															&physicsSystemAllocator,
															&cleanupSystemAllocator,
															&debugSystemAllocator,
															systemData,
															systemHeader->mType,
															&index);
			}
			else
			{
				system = PhysicsEngine::loadSystem(&systemAllocatorMap,
													systemData,
													systemHeader->mType,
													&index);
			}

			assert(system != NULL && "Returned a NULL system after loading\n");
			assert(index >= 0 && "Returned a negative index for system after loading\n");

			idToGlobalIndex[system->getId()] = index;
			idToType[system->getId()] = systemHeader->mType;
		}
	}
}