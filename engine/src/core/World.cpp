#include <fstream>
#include <stack>

#include "../../include/core/Load.h"
#include "../../include/core/LoadInternal.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

World::World()
{
    mPrimitives.createPrimitiveMeshes(this, 10, 10);
}

World::~World()
{
}

void World::loadAssetsInPath(const std::filesystem::path &filePath)
{
    if (std::filesystem::is_directory(filePath))
    {
        std::stack<std::filesystem::path> stack;
        stack.push(filePath);

        while (!stack.empty())
        {
            std::filesystem::path currentPath = stack.top();
            stack.pop();

            std::error_code error_code;
            for (const std::filesystem::directory_entry &entry :
                 std::filesystem::directory_iterator(currentPath, error_code))
            {
                if (std::filesystem::is_directory(entry, error_code))
                {
                    stack.push(entry.path());
                }
                else if (std::filesystem::is_regular_file(entry, error_code))
                {
                    std::string extension = entry.path().extension().string();
                    if (extension == ".mesh" || extension == ".shader" || extension == ".material" ||
                        extension == ".texture")
                    {
                        std::filesystem::path relativeDataPath =
                            entry.path().lexically_relative(std::filesystem::current_path());
                        loadAssetFromYAML(relativeDataPath.string());
                    }
                }
            }
        }
    }
}

Asset *World::loadAssetFromYAML(const std::string &filePath)
{
    YAML::Node in;
    try
    {
        in = YAML::LoadFile(filePath);
    }
    catch (YAML::Exception e /*YAML::BadFile e*/)
    {
        Log::error("YAML exception hit when trying to load file");
        return nullptr;
    }

    if (!in.IsMap() || in.begin() == in.end())
    {
        return nullptr;
    }

    if (in.begin()->first.IsScalar() && in.begin()->second.IsMap())
    {
        Asset *asset = loadAssetFromYAML(in.begin()->second);
        if (asset != nullptr)
        {
            mIdState.mAssetIdToFilepath[asset->getId()] = filePath;
            mIdState.mAssetFilepathToId[filePath] = asset->getId();
        }

        return asset;
    }

    return nullptr;
}

Scene *World::loadSceneFromYAML(const std::string &filePath)
{
    YAML::Node in;
    try
    {
        in = YAML::LoadFile(filePath);
    }
    catch (YAML::BadFile e)
    {
        Log::error("YAML exception hit when trying to load file");
        return nullptr;
    }

    Scene *scene = loadSceneFromYAML(in);
    if (scene != nullptr)
    {
        mIdState.mSceneIdToFilepath[scene->getId()] = filePath;
        mIdState.mSceneFilepathToId[filePath] = scene->getId();
    }

    return scene;
}

bool World::writeAssetToYAML(const std::string &filePath, const Guid &assetId) const
{
    std::ofstream out;
    out.open(filePath);

    if (!out.is_open())
    {
        std::string errorMessage = "Failed to open asset file " + filePath + "\n";
        Log::error(&errorMessage[0]);
        return false;
    }

    int type = getTypeOf(assetId);

    Asset *asset = nullptr;

    if (Asset::isInternal(type))
    {
        asset = PhysicsEngine::getInternalAsset(mAllocators, mIdState, assetId, type);
    }
    else
    {
        asset = PhysicsEngine::getAsset(mAllocators, mIdState, assetId, type);
    }

    if (asset->mHide == HideFlag::None)
    {
        YAML::Node an;
        asset->serialize(an);

        YAML::Node assetNode;
        assetNode[asset->getObjectName()] = an;

        out << assetNode;
        out << "\n";
    }

    out.close();

    return true;
}

bool World::writeSceneToYAML(const std::string &filePath, const Guid &sceneId) const
{
    std::ofstream out;
    out.open(filePath);

    if (!out.is_open())
    {
        std::string errorMessage = "Failed to open scene file " + filePath + "\n";
        Log::error(&errorMessage[0]);
        return false;
    }

    Scene *scene = getSceneById(sceneId);
    if (scene == nullptr)
    {
        return false;
    }

    YAML::Node sceneNode;
    scene->serialize(sceneNode);

    out << sceneNode;
    out << "\n";

    for (size_t i = 0; i < getNumberOfEntities(); i++)
    {
        const Entity *entity = getEntityByIndex(i);

        if (entity->mHide == HideFlag::None)
        {
            YAML::Node en;
            entity->serialize(en);

            YAML::Node entityNode;
            entityNode[entity->getObjectName()] = en;

            out << entityNode;
            out << "\n";

            std::vector<std::pair<Guid, int>> temp = entity->getComponentsOnEntity();
            for (size_t j = 0; j < temp.size(); j++)
            {
                Component *component = nullptr;

                if (Component::isInternal(temp[j].second))
                {
                    component =
                        PhysicsEngine::getInternalComponent(mAllocators, mIdState, temp[j].first, temp[j].second);
                }
                else
                {
                    component = PhysicsEngine::getComponent(mAllocators, mIdState, temp[j].first, temp[j].second);
                }

                if (component->mHide == HideFlag::None)
                {
                    YAML::Node cn;
                    component->serialize(cn);

                    YAML::Node componentNode;
                    componentNode[component->getObjectName()] = cn;

                    out << componentNode;
                    out << "\n";
                }
            }
        }
    }

    out.close();

    return true;
}

Asset *World::loadAssetFromYAML(const YAML::Node &in)
{
    int type = YAML::getValue<int>(in, "type");
    Guid id = YAML::getValue<Guid>(in, "id");

    if (PhysicsEngine::isAsset(type) && id.isValid())
    {
        return loadAssetFromYAML(in, id, type);
    }

    return nullptr;
}

Scene *World::loadSceneFromYAML(const YAML::Node &in)
{
    int type = YAML::getValue<int>(in, "type");
    Guid id = YAML::getValue<Guid>(in, "id");

    if (PhysicsEngine::isScene(type) && id.isValid())
    {
        return loadSceneFromYAML(in, id);
    }

    return nullptr;
}

Asset *World::loadAssetFromYAML(const YAML::Node &in, const Guid id, int type)
{
    if (Asset::isInternal(type))
    {
        return PhysicsEngine::loadInternalAsset(*this, mAllocators, mIdState, in, id, type);
    }
    else
    {
        return PhysicsEngine::loadAsset(*this, mAllocators, mIdState, in, id, type);
    }
}

Scene *World::loadSceneFromYAML(const YAML::Node &in, const Guid id)
{
    return PhysicsEngine::loadInternalScene(*this, mAllocators, mIdState, in, id);
}

void World::latentDestroyEntitiesInWorld()
{
    // latent destroy all entities (and thereby also all components)
    for (size_t i = 0; i < getNumberOfEntities(); i++)
    {
        Entity *entity = getEntityByIndex(i);

        if (!entity->mDoNotDestroy)
        {
            latentDestroyEntity(entity->getId());
        }
    }
}

void World::immediateDestroyEntitiesInWorld()
{
    // immediate destroy all entities (and thereby also all components)
    std::vector<Guid> entitiesToDestroy;
    for (size_t i = 0; i < getNumberOfEntities(); i++)
    {
        Entity *entity = getEntityByIndex(i);

        if (!entity->mDoNotDestroy)
        {
            entitiesToDestroy.push_back(entity->getId());
        }
    }

    for (size_t i = 0; i < entitiesToDestroy.size(); i++)
    {
        Log::info(("Immediate destroy entity with id: " + entitiesToDestroy[i].toString() + "\n").c_str());
        immediateDestroyEntity(entitiesToDestroy[i]);
    }
}

std::vector<ShaderUniform> World::getCachedMaterialUniforms(const Guid &materialId, const Guid &shaderId)
{
    return mMaterialUniformCache[materialId][shaderId];
}

void World::cacheMaterialUniforms(const Guid &materialId, const Guid &shaderId, const std::vector<ShaderUniform> &uniforms)
{
    assert(materialId != Guid::INVALID);
    assert(shaderId != Guid::INVALID);

    mMaterialUniformCache[materialId][shaderId] = uniforms;
}

size_t World::getNumberOfScenes() const
{
    return mAllocators.mSceneAllocator.getCount();
}

size_t World::getNumberOfEntities() const
{
    return mAllocators.mEntityAllocator.getCount();
}

size_t World::getNumberOfNonHiddenEntities() const
{
    size_t count = 0;
    for (size_t i = 0; i < getNumberOfEntities(); i++)
    {
        const Entity *entity = getEntityByIndex(i);
        if (entity->mHide == HideFlag::None)
        {
            count++;
        }
    }

    return count;
}

size_t World::getNumberOfUpdatingSystems() const
{
    return mSystems.size();
}

Mesh *World::getPrimtiveMesh(PrimitiveType type) const
{
    switch (type)
    {
    case PrimitiveType::Plane:
        return getAssetById<Mesh>(mPrimitives.mPlaneMeshId);
    case PrimitiveType::Disc:
        return getAssetById<Mesh>(mPrimitives.mDiscMeshId);
    case PrimitiveType::Cube:
        return getAssetById<Mesh>(mPrimitives.mCubeMeshId);
    case PrimitiveType::Sphere:
        return getAssetById<Mesh>(mPrimitives.mSphereMeshId);
    case PrimitiveType::Cylinder:
        return getAssetById<Mesh>(mPrimitives.mCylinderMeshId);
    case PrimitiveType::Cone:
        return getAssetById<Mesh>(mPrimitives.mConeMeshId);
    default:
        return nullptr;
    }
}

Entity *World::createPrimitive(PrimitiveType type)
{
    Mesh *mesh = getPrimtiveMesh(type);
    Entity *entity = createEntity();
    Transform* transform = entity->addComponent<Transform>();
    MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>();
    
    assert(mesh != nullptr);
    assert(entity != nullptr);
    assert(transform != nullptr);
    assert(meshRenderer != nullptr);

    entity->setName(mesh->getName());
    
    transform->mPosition = glm::vec3(0, 0, 0);
    transform->mRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    transform->mScale = glm::vec3(1, 1, 1);
    meshRenderer->setMesh(mesh->getId());
    meshRenderer->setMaterial(mPrimitives.mStandardMaterialId);

    return entity;
}

Entity *World::createNonPrimitive(const Guid &meshId)
{
    Mesh *mesh = getAssetById<Mesh>(meshId);
    if (mesh == nullptr)
    {
        return nullptr;
    }

    Entity *entity = createEntity();
    Transform *transform = entity->addComponent<Transform>();
    MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>();

    assert(entity != nullptr);
    assert(transform != nullptr);
    assert(meshRenderer != nullptr);

    entity->setName(mesh->getName());
   
    transform->mPosition = glm::vec3(0, 0, 0);
    transform->mRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    transform->mScale = glm::vec3(1, 1, 1);
    meshRenderer->setMesh(meshId);
    meshRenderer->setMaterial(mPrimitives.mStandardMaterialId);

    return entity;
}

Entity *World::createLight(LightType type)
{
    Entity *entity = createEntity();
    entity->addComponent<Transform>();
    Light* light = entity->addComponent<Light>();

    switch (type)
    {
    case LightType::Directional:
        light->mLightType = LightType::Directional;
        break;
    case LightType::Spot:
        light->mLightType = LightType::Spot;
        break;
    case LightType::Point:
        light->mLightType = LightType::Point;
        break;
    }

    return entity;
}

Entity *World::createCamera()
{
    Entity *entity = createEntity();
    entity->addComponent<Transform>();
    entity->addComponent<Camera>();

    return entity;
}

Scene *World::getSceneById(const Guid &sceneId) const
{
    return getById_impl<Scene>(mIdState.mSceneIdToGlobalIndex, &mAllocators.mSceneAllocator, sceneId);
}

Scene *World::getSceneByIndex(size_t index) const
{
    return mAllocators.mSceneAllocator.get(index);
}

Entity *World::getEntityById(const Guid &entityId) const
{
    return getById_impl<Entity>(mIdState.mEntityIdToGlobalIndex, &mAllocators.mEntityAllocator, entityId);
}

Entity *World::getEntityByIndex(size_t index) const
{
    return mAllocators.mEntityAllocator.get(index);
}

System *World::getSystemByUpdateOrder(size_t order) const
{
    if (order >= mSystems.size())
    {
        return nullptr;
    }

    return mSystems[order];
}

int World::getIndexOf(const Guid &id) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mIdToGlobalIndex.find(id);
    if (it != mIdState.mIdToGlobalIndex.end())
    {
        return it->second;
    }

    return -1;
}

int World::getTypeOf(const Guid &id) const
{
    std::unordered_map<Guid, int>::const_iterator it = mIdState.mIdToType.find(id);
    if (it != mIdState.mIdToType.end())
    {
        return it->second;
    }

    return -1;
}

Scene *World::createScene()
{
    int globalIndex = (int)mAllocators.mSceneAllocator.getCount();
    int type = SceneType<Scene>::type;
    Guid sceneId = Guid::newGuid();

    Scene *scene = mAllocators.mSceneAllocator.construct(this, sceneId);

    if (scene != nullptr)
    {
        addIdToGlobalIndexMap_impl<Scene>(scene->getId(), globalIndex, type);
    }

    return scene;
}

Entity *World::createEntity()
{
    int globalIndex = (int)mAllocators.mEntityAllocator.getCount();
    int type = EntityType<Entity>::type;
    Guid entityId = Guid::newGuid();

    Entity *entity = mAllocators.mEntityAllocator.construct(this, entityId);

    if (entity != nullptr)
    {
        addIdToGlobalIndexMap_impl<Entity>(entity->getId(), globalIndex, type);

        mIdState.mEntityIdToComponentIds[entityId] = std::vector<std::pair<Guid, int>>();

        mIdState.mEntityIdsMarkedCreated.push_back(entityId);
    }

    return entity;
}

void World::latentDestroyEntity(const Guid &entityId)
{
    mIdState.mEntityIdsMarkedLatentDestroy.push_back(entityId);

    // add any components found on the entity to the latent destroy component list
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::const_iterator it =
        mIdState.mEntityIdToComponentIds.find(entityId);

    assert(it != mIdState.mEntityIdToComponentIds.end());

    for (size_t i = 0; i < it->second.size(); i++)
    {
        latentDestroyComponent(entityId, it->second[i].first, it->second[i].second);
    }
}

void World::immediateDestroyEntity(const Guid &entityId)
{
    std::vector<std::pair<Guid, int>> componentsOnEntity = mIdState.mEntityIdToComponentIds[entityId];
    for (size_t i = 0; i < componentsOnEntity.size(); i++)
    {
        immediateDestroyComponent(entityId, componentsOnEntity[i].first, componentsOnEntity[i].second);
    }

    assert(mIdState.mEntityIdToComponentIds[entityId].size() == 0);

    mIdState.mEntityIdToComponentIds.erase(entityId);

    destroyInternalEntity(mAllocators, mIdState, entityId, getIndexOf(entityId));
}

void World::latentDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType)
{
    mIdState.mComponentIdsMarkedLatentDestroy.push_back(std::make_tuple(entityId, componentId, componentType));
}

void World::immediateDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType)
{
    // remove from entity component list
    std::vector<std::pair<Guid, int>> &componentsOnEntity = mIdState.mEntityIdToComponentIds[entityId];

    std::vector<std::pair<Guid, int>>::iterator it = componentsOnEntity.begin();
    while (it < componentsOnEntity.end())
    {
        if (it->second == componentType && it->first == componentId)
        {
            break;
        }

        it++;
    }

    if (it < componentsOnEntity.end())
    {
        componentsOnEntity.erase(it);
    }

    if (Component::isInternal(componentType))
    {
        destroyInternalComponent(mAllocators, mIdState, entityId, componentId, componentType, getIndexOf(componentId));
    }
    else
    {
        destroyComponent(mAllocators, mIdState, entityId, componentId, componentType, getIndexOf(componentId));
    }
}

void World::latentDestroyAsset(const Guid &assetId, int assetType)
{
    mIdState.mAssetIdsMarkedLatentDestroy.push_back(std::make_pair(assetId, assetType));
}

void World::immediateDestroyAsset(const Guid &assetId, int assetType)
{
    if (Asset::isInternal(assetType))
    {
        destroyInternalAsset(mAllocators, mIdState, assetId, assetType, getIndexOf(assetId));
    }
    else
    {
        destroyAsset(mAllocators, mIdState, assetId, assetType, getIndexOf(assetId));
    }
}

bool World::isMarkedForLatentDestroy(const Guid &id)
{
    for (size_t i = 0; i < mIdState.mEntityIdsMarkedLatentDestroy.size(); i++)
    {
        if (mIdState.mEntityIdsMarkedLatentDestroy[i] == id)
        {
            return true;
        }
    }

    for (size_t i = 0; i < mIdState.mComponentIdsMarkedLatentDestroy.size(); i++)
    {
        if (std::get<1>(mIdState.mComponentIdsMarkedLatentDestroy[i]) == id)
        {
            return true;
        }
    }

    return false;
}

void World::clearIdsMarkedCreatedOrDestroyed()
{
    mIdState.mEntityIdsMarkedCreated.clear();
    mIdState.mEntityIdsMarkedLatentDestroy.clear();
    mIdState.mComponentIdsMarkedCreated.clear();
    mIdState.mComponentIdsMarkedLatentDestroy.clear();
}

std::vector<std::pair<Guid, int>> World::getComponentsOnEntity(const Guid &entityId) const
{
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>::const_iterator it =
        mIdState.mEntityIdToComponentIds.find(entityId);
    if (it != mIdState.mEntityIdToComponentIds.end())
    {
        return it->second;
    }

    return std::vector<std::pair<Guid, int>>();
}

std::vector<Guid> World::getEntityIdsMarkedCreated() const
{
    return mIdState.mEntityIdsMarkedCreated;
}

std::vector<Guid> World::getEntityIdsMarkedLatentDestroy() const
{
    return mIdState.mEntityIdsMarkedLatentDestroy;
}

std::vector<std::tuple<Guid, Guid, int>> World::getComponentIdsMarkedCreated() const
{
    return mIdState.mComponentIdsMarkedCreated;
}

std::vector<std::tuple<Guid, Guid, int>> World::getComponentIdsMarkedLatentDestroy() const
{
    return mIdState.mComponentIdsMarkedLatentDestroy;
}

std::string World::getAssetFilepath(const Guid &assetId) const
{
    std::unordered_map<Guid, std::string>::const_iterator it = mIdState.mAssetIdToFilepath.find(assetId);
    if (it != mIdState.mAssetIdToFilepath.end())
    {
        return it->second;
    }

    return std::string();
}

std::string World::getSceneFilepath(const Guid &sceneId) const
{
    std::unordered_map<Guid, std::string>::const_iterator it = mIdState.mSceneIdToFilepath.find(sceneId);
    if (it != mIdState.mSceneIdToFilepath.end())
    {
        return it->second;
    }

    return std::string();
}

Guid World::getAssetId(const std::string& filepath) const
{
    std::unordered_map<std::string, Guid>::const_iterator it = mIdState.mAssetFilepathToId.find(filepath);
    if (it != mIdState.mAssetFilepathToId.end())
    {
        return it->second;
    }

    return Guid::INVALID;
}

Guid World::getSceneId(const std::string& filepath) const
{
    std::unordered_map<std::string, Guid>::const_iterator it = mIdState.mSceneFilepathToId.find(filepath);
    if (it != mIdState.mSceneFilepathToId.end())
    {
        return it->second;
    }

    return Guid::INVALID;
}

// bool World::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance)
//{
//	Ray ray;
//
//	ray.origin = origin;
//	ray.direction = direction;
//
//	return sgrid.intersect(ray) != NULL;// || dtree.intersect(ray) != NULL;
//	// return stree.intersect(ray) != NULL || dtree.intersect(ray) != NULL;
//}
//
//// begin by only implementing for spheres first and later I will add for bounds, capsules etc
// bool World::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance, Collider** collider)
//{
//	Ray ray;
//
//	ray.origin = origin;
//	ray.direction = direction;
//
//	// Object* object = stree.intersect(ray);
//	BoundingSphere* boundingSphere = sgrid.intersect(ray);
//
//	if(boundingSphere != NULL){
//		//std::cout << "AAAAAA id: " << boundingSphere->id.toString() << std::endl;
//		std::map<Guid, int>::iterator it = idToGlobalIndex.find(boundingSphere->id);
//		if(it != idToGlobalIndex.end()){
//			int colliderIndex = it->second;
//
//			if(boundingSphere->primitiveType == 0){
//				*collider = getComponentByIndex<SphereCollider>(colliderIndex);
//			}
//			else if(boundingSphere->primitiveType == 1){
//				*collider = getComponentByIndex<BoxCollider>(colliderIndex);
//			}
//			else{
//				*collider = getComponentByIndex<MeshCollider>(colliderIndex);
//			}
//			return true;
//		}
//		else{
//			std::cout << "Error: component id does not correspond to a global index" << std::endl;
//			return false;
//		}
//	}
//
//	return false;
//}

// Explicit template specializations

template <> size_t World::getNumberOfSystems<RenderSystem>() const
{
    return mAllocators.mRenderSystemAllocator.getCount();
}

template <> size_t World::getNumberOfSystems<PhysicsSystem>() const
{
    return mAllocators.mPhysicsSystemAllocator.getCount();
}

template <> size_t World::getNumberOfSystems<CleanUpSystem>() const
{
    return mAllocators.mCleanupSystemAllocator.getCount();
}

template <> size_t World::getNumberOfSystems<DebugSystem>() const
{
    return mAllocators.mDebugSystemAllocator.getCount();
}

template <> size_t World::getNumberOfSystems<GizmoSystem>() const
{
    return mAllocators.mGizmoSystemAllocator.getCount();
}

template <> size_t World::getNumberOfSystems<FreeLookCameraSystem>() const
{
    return mAllocators.mFreeLookCameraSystemAllocator.getCount();
}

template <> size_t World::getNumberOfComponents<Transform>() const
{
    return mAllocators.mTransformAllocator.getCount();
}

template <> size_t World::getNumberOfComponents<MeshRenderer>() const
{
    return mAllocators.mMeshRendererAllocator.getCount();
}

template <> size_t World::getNumberOfComponents<SpriteRenderer>() const
{
    return mAllocators.mSpriteRendererAllocator.getCount();
}

template <> size_t World::getNumberOfComponents<LineRenderer>() const
{
    return mAllocators.mLineRendererAllocator.getCount();
}

template <> size_t World::getNumberOfComponents<Rigidbody>() const
{
    return mAllocators.mRigidbodyAllocator.getCount();
}

template <> size_t World::getNumberOfComponents<Camera>() const
{
    return mAllocators.mCameraAllocator.getCount();
}

template <> size_t World::getNumberOfComponents<Light>() const
{
    return mAllocators.mLightAllocator.getCount();
}

template <> size_t World::getNumberOfComponents<SphereCollider>() const
{
    return mAllocators.mSphereColliderAllocator.getCount();
}

template <> size_t World::getNumberOfComponents<BoxCollider>() const
{
    return mAllocators.mBoxColliderAllocator.getCount();
}

template <> size_t World::getNumberOfComponents<CapsuleCollider>() const
{
    return mAllocators.mCapsuleColliderAllocator.getCount();
}

template <> size_t World::getNumberOfComponents<MeshCollider>() const
{
    return mAllocators.mMeshColliderAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Mesh>() const
{
    return mAllocators.mMeshAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Material>() const
{
    return mAllocators.mMaterialAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Shader>() const
{
    return mAllocators.mShaderAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Texture2D>() const
{
    return mAllocators.mTexture2DAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Texture3D>() const
{
    return mAllocators.mTexture3DAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Cubemap>() const
{
    return mAllocators.mCubemapAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<RenderTexture>() const
{
    return mAllocators.mRenderTextureAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Font>() const
{
    return mAllocators.mFontAllocator.getCount();
}

template <> size_t World::getNumberOfAssets<Sprite>() const
{
    return mAllocators.mSpriteAllocator.getCount();
}

template <> RenderSystem *World::getSystem<RenderSystem>() const
{
    return getSystem_impl(&mAllocators.mRenderSystemAllocator);
}

template <> PhysicsSystem *World::getSystem<PhysicsSystem>() const
{
    return getSystem_impl(&mAllocators.mPhysicsSystemAllocator);
}

template <> CleanUpSystem *World::getSystem<CleanUpSystem>() const
{
    return getSystem_impl(&mAllocators.mCleanupSystemAllocator);
}

template <> DebugSystem *World::getSystem<DebugSystem>() const
{
    return getSystem_impl(&mAllocators.mDebugSystemAllocator);
}

template <> GizmoSystem *World::getSystem<GizmoSystem>() const
{
    return getSystem_impl(&mAllocators.mGizmoSystemAllocator);
}

template <> FreeLookCameraSystem *World::getSystem<FreeLookCameraSystem>() const
{
    return getSystem_impl(&mAllocators.mFreeLookCameraSystemAllocator);
}

template <> Transform *World::getComponent<Transform>(const Guid &entityId) const
{
    return getComponent_impl(&mAllocators.mTransformAllocator, entityId);
}

template <> MeshRenderer *World::getComponent<MeshRenderer>(const Guid &entityId) const
{
    return getComponent_impl(&mAllocators.mMeshRendererAllocator, entityId);
}

template <> SpriteRenderer *World::getComponent<SpriteRenderer>(const Guid &entityId) const
{
    return getComponent_impl(&mAllocators.mSpriteRendererAllocator, entityId);
}

template <> LineRenderer *World::getComponent<LineRenderer>(const Guid &entityId) const
{
    return getComponent_impl(&mAllocators.mLineRendererAllocator, entityId);
}

template <> Rigidbody *World::getComponent<Rigidbody>(const Guid &entityId) const
{
    return getComponent_impl(&mAllocators.mRigidbodyAllocator, entityId);
}

template <> Camera *World::getComponent<Camera>(const Guid &entityId) const
{
    return getComponent_impl(&mAllocators.mCameraAllocator, entityId);
}

template <> Light *World::getComponent<Light>(const Guid &entityId) const
{
    return getComponent_impl(&mAllocators.mLightAllocator, entityId);
}

template <> SphereCollider *World::getComponent<SphereCollider>(const Guid &entityId) const
{
    return getComponent_impl(&mAllocators.mSphereColliderAllocator, entityId);
}

template <> BoxCollider *World::getComponent<BoxCollider>(const Guid &entityId) const
{
    return getComponent_impl(&mAllocators.mBoxColliderAllocator, entityId);
}

template <> CapsuleCollider *World::getComponent<CapsuleCollider>(const Guid &entityId) const
{
    return getComponent_impl(&mAllocators.mCapsuleColliderAllocator, entityId);
}

template <> MeshCollider *World::getComponent<MeshCollider>(const Guid &entityId) const
{
    return getComponent_impl(&mAllocators.mMeshColliderAllocator, entityId);
}

template <> Transform *World::addComponent<Transform>(const Guid &entityId)
{
    return addComponent_impl(&mAllocators.mTransformAllocator, entityId);
}

template <> MeshRenderer *World::addComponent<MeshRenderer>(const Guid &entityId)
{
    return addComponent_impl(&mAllocators.mMeshRendererAllocator, entityId);
}

template <> SpriteRenderer *World::addComponent<SpriteRenderer>(const Guid &entityId)
{
    return addComponent_impl(&mAllocators.mSpriteRendererAllocator, entityId);
}

template <> LineRenderer *World::addComponent<LineRenderer>(const Guid &entityId)
{
    return addComponent_impl(&mAllocators.mLineRendererAllocator, entityId);
}

template <> Rigidbody *World::addComponent<Rigidbody>(const Guid &entityId)
{
    return addComponent_impl(&mAllocators.mRigidbodyAllocator, entityId);
}

template <> Camera *World::addComponent<Camera>(const Guid &entityId)
{
    return addComponent_impl(&mAllocators.mCameraAllocator, entityId);
}

template <> Light *World::addComponent<Light>(const Guid &entityId)
{
    return addComponent_impl(&mAllocators.mLightAllocator, entityId);
}

template <> SphereCollider *World::addComponent<SphereCollider>(const Guid &entityId)
{
    return addComponent_impl(&mAllocators.mSphereColliderAllocator, entityId);
}

template <> BoxCollider *World::addComponent<BoxCollider>(const Guid &entityId)
{
    return addComponent_impl(&mAllocators.mBoxColliderAllocator, entityId);
}

template <> CapsuleCollider *World::addComponent<CapsuleCollider>(const Guid &entityId)
{
    return addComponent_impl(&mAllocators.mCapsuleColliderAllocator, entityId);
}

template <> MeshCollider *World::addComponent<MeshCollider>(const Guid &entityId)
{
    return addComponent_impl(&mAllocators.mMeshColliderAllocator, entityId);
}

template <> RenderSystem *World::addSystem<RenderSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mRenderSystemAllocator, order);
}

template <> PhysicsSystem *World::addSystem<PhysicsSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mPhysicsSystemAllocator, order);
}

template <> CleanUpSystem *World::addSystem<CleanUpSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mCleanupSystemAllocator, order);
}

template <> DebugSystem *World::addSystem<DebugSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mDebugSystemAllocator, order);
}

template <> GizmoSystem *World::addSystem<GizmoSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mGizmoSystemAllocator, order);
}

template <> FreeLookCameraSystem *World::addSystem<FreeLookCameraSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mFreeLookCameraSystemAllocator, order);
}

template <> RenderSystem *World::getSystemByIndex<RenderSystem>(size_t index) const
{
    return getSystemByIndex_impl(&mAllocators.mRenderSystemAllocator, index);
}

template <> PhysicsSystem *World::getSystemByIndex<PhysicsSystem>(size_t index) const
{
    return getSystemByIndex_impl(&mAllocators.mPhysicsSystemAllocator, index);
}

template <> CleanUpSystem *World::getSystemByIndex<CleanUpSystem>(size_t index) const
{
    return getSystemByIndex_impl(&mAllocators.mCleanupSystemAllocator, index);
}

template <> DebugSystem *World::getSystemByIndex<DebugSystem>(size_t index) const
{
    return getSystemByIndex_impl(&mAllocators.mDebugSystemAllocator, index);
}

template <> GizmoSystem *World::getSystemByIndex<GizmoSystem>(size_t index) const
{
    return getSystemByIndex_impl(&mAllocators.mGizmoSystemAllocator, index);
}

template <> FreeLookCameraSystem *World::getSystemByIndex<FreeLookCameraSystem>(size_t index) const
{
    return getSystemByIndex_impl(&mAllocators.mFreeLookCameraSystemAllocator, index);
}

template <> RenderSystem *World::getSystemById<RenderSystem>(const Guid &systemId) const
{
    return getSystemById_impl(&mAllocators.mRenderSystemAllocator, systemId);
}

template <> PhysicsSystem *World::getSystemById<PhysicsSystem>(const Guid &systemId) const
{
    return getSystemById_impl(&mAllocators.mPhysicsSystemAllocator, systemId);
}

template <> CleanUpSystem *World::getSystemById<CleanUpSystem>(const Guid &systemId) const
{
    return getSystemById_impl(&mAllocators.mCleanupSystemAllocator, systemId);
}

template <> DebugSystem *World::getSystemById<DebugSystem>(const Guid &systemId) const
{
    return getSystemById_impl(&mAllocators.mDebugSystemAllocator, systemId);
}

template <> GizmoSystem *World::getSystemById<GizmoSystem>(const Guid &systemId) const
{
    return getSystemById_impl(&mAllocators.mGizmoSystemAllocator, systemId);
}

template <> FreeLookCameraSystem *World::getSystemById<FreeLookCameraSystem>(const Guid &systemId) const
{
    return getSystemById_impl(&mAllocators.mFreeLookCameraSystemAllocator, systemId);
}

template <> Mesh *World::getAssetByIndex<Mesh>(size_t index) const
{
    return getAssetByIndex_impl(&mAllocators.mMeshAllocator, index);
}

template <> Material *World::getAssetByIndex<Material>(size_t index) const
{
    return getAssetByIndex_impl(&mAllocators.mMaterialAllocator, index);
}

template <> Shader *World::getAssetByIndex<Shader>(size_t index) const
{
    return getAssetByIndex_impl(&mAllocators.mShaderAllocator, index);
}

template <> Texture2D *World::getAssetByIndex<Texture2D>(size_t index) const
{
    return getAssetByIndex_impl(&mAllocators.mTexture2DAllocator, index);
}

template <> Texture3D *World::getAssetByIndex<Texture3D>(size_t index) const
{
    return getAssetByIndex_impl(&mAllocators.mTexture3DAllocator, index);
}

template <> Cubemap *World::getAssetByIndex<Cubemap>(size_t index) const
{
    return getAssetByIndex_impl(&mAllocators.mCubemapAllocator, index);
}

template <> RenderTexture *World::getAssetByIndex<RenderTexture>(size_t index) const
{
    return getAssetByIndex_impl(&mAllocators.mRenderTextureAllocator, index);
}

template <> Font *World::getAssetByIndex<Font>(size_t index) const
{
    return getAssetByIndex_impl(&mAllocators.mFontAllocator, index);
}

template <> Sprite *World::getAssetByIndex<Sprite>(size_t index) const
{
    return getAssetByIndex_impl(&mAllocators.mSpriteAllocator, index);
}

template <> Mesh *World::getAssetById<Mesh>(const Guid &assetId) const
{
    return getAssetById_impl(&mAllocators.mMeshAllocator, assetId);
}

template <> Material *World::getAssetById<Material>(const Guid &assetId) const
{
    return getAssetById_impl(&mAllocators.mMaterialAllocator, assetId);
}

template <> Shader *World::getAssetById<Shader>(const Guid &assetId) const
{
    return getAssetById_impl(&mAllocators.mShaderAllocator, assetId);
}

template <> Texture2D *World::getAssetById<Texture2D>(const Guid &assetId) const
{
    return getAssetById_impl(&mAllocators.mTexture2DAllocator, assetId);
}

template <> Texture3D *World::getAssetById<Texture3D>(const Guid &assetId) const
{
    return getAssetById_impl(&mAllocators.mTexture3DAllocator, assetId);
}

template <> Cubemap *World::getAssetById<Cubemap>(const Guid &assetId) const
{
    return getAssetById_impl(&mAllocators.mCubemapAllocator, assetId);
}

template <> RenderTexture *World::getAssetById<RenderTexture>(const Guid &assetId) const
{
    return getAssetById_impl(&mAllocators.mRenderTextureAllocator, assetId);
}

template <> Font *World::getAssetById<Font>(const Guid &assetId) const
{
    return getAssetById_impl(&mAllocators.mFontAllocator, assetId);
}

template <> Sprite *World::getAssetById<Sprite>(const Guid &assetId) const
{
    return getAssetById_impl(&mAllocators.mSpriteAllocator, assetId);
}

template <> Transform *World::getComponentByIndex<Transform>(size_t index) const
{
    return getComponentByIndex_impl(&mAllocators.mTransformAllocator, index);
}

template <> MeshRenderer *World::getComponentByIndex<MeshRenderer>(size_t index) const
{
    return getComponentByIndex_impl(&mAllocators.mMeshRendererAllocator, index);
}

template <> SpriteRenderer *World::getComponentByIndex<SpriteRenderer>(size_t index) const
{
    return getComponentByIndex_impl(&mAllocators.mSpriteRendererAllocator, index);
}

template <> LineRenderer *World::getComponentByIndex<LineRenderer>(size_t index) const
{
    return getComponentByIndex_impl(&mAllocators.mLineRendererAllocator, index);
}

template <> Rigidbody *World::getComponentByIndex<Rigidbody>(size_t index) const
{
    return getComponentByIndex_impl(&mAllocators.mRigidbodyAllocator, index);
}

template <> Camera *World::getComponentByIndex<Camera>(size_t index) const
{
    return getComponentByIndex_impl(&mAllocators.mCameraAllocator, index);
}

template <> Light *World::getComponentByIndex<Light>(size_t index) const
{
    return getComponentByIndex_impl(&mAllocators.mLightAllocator, index);
}

template <> SphereCollider *World::getComponentByIndex<SphereCollider>(size_t index) const
{
    return getComponentByIndex_impl(&mAllocators.mSphereColliderAllocator, index);
}

template <> BoxCollider *World::getComponentByIndex<BoxCollider>(size_t index) const
{
    return getComponentByIndex_impl(&mAllocators.mBoxColliderAllocator, index);
}

template <> CapsuleCollider *World::getComponentByIndex<CapsuleCollider>(size_t index) const
{
    return getComponentByIndex_impl(&mAllocators.mCapsuleColliderAllocator, index);
}

template <> MeshCollider *World::getComponentByIndex<MeshCollider>(size_t index) const
{
    return getComponentByIndex_impl(&mAllocators.mMeshColliderAllocator, index);
}

template <> Transform *World::getComponentById<Transform>(const Guid &componentId) const
{
    return getComponentById_impl(&mAllocators.mTransformAllocator, componentId);
}

template <> MeshRenderer *World::getComponentById<MeshRenderer>(const Guid &componentId) const
{
    return getComponentById_impl(&mAllocators.mMeshRendererAllocator, componentId);
}

template <> SpriteRenderer *World::getComponentById<SpriteRenderer>(const Guid &componentId) const
{
    return getComponentById_impl(&mAllocators.mSpriteRendererAllocator, componentId);
}

template <> LineRenderer *World::getComponentById<LineRenderer>(const Guid &componentId) const
{
    return getComponentById_impl(&mAllocators.mLineRendererAllocator, componentId);
}

template <> Rigidbody *World::getComponentById<Rigidbody>(const Guid &componentId) const
{
    return getComponentById_impl(&mAllocators.mRigidbodyAllocator, componentId);
}

template <> Camera *World::getComponentById<Camera>(const Guid &componentId) const
{
    return getComponentById_impl(&mAllocators.mCameraAllocator, componentId);
}

template <> Light *World::getComponentById<Light>(const Guid &componentId) const
{
    return getComponentById_impl(&mAllocators.mLightAllocator, componentId);
}

template <> SphereCollider *World::getComponentById<SphereCollider>(const Guid &componentId) const
{
    return getComponentById_impl(&mAllocators.mSphereColliderAllocator, componentId);
}

template <> BoxCollider *World::getComponentById<BoxCollider>(const Guid &componentId) const
{
    return getComponentById_impl(&mAllocators.mBoxColliderAllocator, componentId);
}

template <> CapsuleCollider *World::getComponentById<CapsuleCollider>(const Guid &componentId) const
{
    return getComponentById_impl(&mAllocators.mCapsuleColliderAllocator, componentId);
}

template <> MeshCollider *World::getComponentById<MeshCollider>(const Guid &componentId) const
{
    return getComponentById_impl(&mAllocators.mMeshColliderAllocator, componentId);
}

template <> Mesh *World::createAsset<Mesh>()
{
    return createAsset_impl(&mAllocators.mMeshAllocator, Guid::newGuid());
}

template <> Mesh *World::createAsset<Mesh>(const Guid &id)
{
    return createAsset_impl(&mAllocators.mMeshAllocator, id);
}

template <> Material *World::createAsset<Material>()
{
    return createAsset_impl(&mAllocators.mMaterialAllocator, Guid::newGuid());
}

template <> Material *World::createAsset<Material>(const Guid &id)
{
    return createAsset_impl(&mAllocators.mMaterialAllocator, id);
}

template <> Shader *World::createAsset<Shader>()
{
    return createAsset_impl(&mAllocators.mShaderAllocator, Guid::newGuid());
}

template <> Shader *World::createAsset<Shader>(const Guid &id)
{
    return createAsset_impl(&mAllocators.mShaderAllocator, id);
}

template <> Texture2D *World::createAsset<Texture2D>()
{
    return createAsset_impl(&mAllocators.mTexture2DAllocator, Guid::newGuid());
}

template <> Texture2D *World::createAsset<Texture2D>(const Guid &id)
{
    return createAsset_impl(&mAllocators.mTexture2DAllocator, id);
}

template <> Texture3D *World::createAsset<Texture3D>()
{
    return createAsset_impl(&mAllocators.mTexture3DAllocator, Guid::newGuid());
}

template <> Texture3D *World::createAsset<Texture3D>(const Guid &id)
{
    return createAsset_impl(&mAllocators.mTexture3DAllocator, id);
}

template <> Cubemap *World::createAsset<Cubemap>()
{
    return createAsset_impl(&mAllocators.mCubemapAllocator, Guid::newGuid());
}

template <> Cubemap *World::createAsset<Cubemap>(const Guid &id)
{
    return createAsset_impl(&mAllocators.mCubemapAllocator, id);
}

template <> RenderTexture *World::createAsset<RenderTexture>()
{
    return createAsset_impl(&mAllocators.mRenderTextureAllocator, Guid::newGuid());
}

template <> RenderTexture *World::createAsset<RenderTexture>(const Guid &id)
{
    return createAsset_impl(&mAllocators.mRenderTextureAllocator, id);
}

template <> Font *World::createAsset<Font>()
{
    return createAsset_impl(&mAllocators.mFontAllocator, Guid::newGuid());
}

template <> Font *World::createAsset<Font>(const Guid &id)
{
    return createAsset_impl(&mAllocators.mFontAllocator, id);
}

template <> Sprite *World::createAsset<Sprite>()
{
    return createAsset_impl(&mAllocators.mSpriteAllocator, Guid::newGuid());
}

template <> Sprite *World::createAsset<Sprite>(const Guid &id)
{
    return createAsset_impl(&mAllocators.mSpriteAllocator, id);
}

template <>
Transform *World::getComponentById_impl<Transform>(const PoolAllocator<Transform> *allocator,
                                                   const Guid &componentId) const
{
    return getById_impl<Transform>(mIdState.mTransformIdToGlobalIndex, allocator, componentId);
}

template <>
MeshRenderer *World::getComponentById_impl<MeshRenderer>(const PoolAllocator<MeshRenderer> *allocator,
                                                  const Guid &componentId) const
{
    return getById_impl<MeshRenderer>(mIdState.mMeshRendererIdToGlobalIndex, allocator, componentId);
}

template <>
SpriteRenderer *World::getComponentById_impl<SpriteRenderer>(const PoolAllocator<SpriteRenderer> *allocator,
                                                      const Guid &componentId) const
{
    return getById_impl<SpriteRenderer>(mIdState.mSpriteRendererIdToGlobalIndex, allocator, componentId);
}

template <>
LineRenderer *World::getComponentById_impl<LineRenderer>(const PoolAllocator<LineRenderer> *allocator,
                                                  const Guid &componentId) const
{
    return getById_impl<LineRenderer>(mIdState.mLineRendererIdToGlobalIndex, allocator, componentId);
}

template <>
Rigidbody *World::getComponentById_impl<Rigidbody>(const PoolAllocator<Rigidbody> *allocator,
                                                   const Guid &componentId) const
{
    return getById_impl<Rigidbody>(mIdState.mRigidbodyIdToGlobalIndex, allocator, componentId);
}

template <>
Camera *World::getComponentById_impl<Camera>(const PoolAllocator<Camera> *allocator, const Guid &componentId) const
{
    return getById_impl<Camera>(mIdState.mCameraIdToGlobalIndex, allocator, componentId);
}

template <>
Light *World::getComponentById_impl<Light>(const PoolAllocator<Light> *allocator, const Guid &componentId) const
{
    return getById_impl<Light>(mIdState.mLightIdToGlobalIndex, allocator, componentId);
}

template <>
SphereCollider *World::getComponentById_impl<SphereCollider>(const PoolAllocator<SphereCollider> *allocator,
                                                      const Guid &componentId) const
{
    return getById_impl<SphereCollider>(mIdState.mSphereColliderIdToGlobalIndex, allocator, componentId);
}

template <>
BoxCollider *World::getComponentById_impl<BoxCollider>(const PoolAllocator<BoxCollider> *allocator,
                                                const Guid &componentId) const
{
    return getById_impl<BoxCollider>(mIdState.mBoxColliderIdToGlobalIndex, allocator, componentId);
}

template <>
CapsuleCollider *World::getComponentById_impl<CapsuleCollider>(const PoolAllocator<CapsuleCollider> *allocator,
                                                        const Guid &componentId) const
{
    return getById_impl<CapsuleCollider>(mIdState.mCapsuleColliderIdToGlobalIndex, allocator, componentId);
}

template <>
MeshCollider *World::getComponentById_impl<MeshCollider>(const PoolAllocator<MeshCollider> *allocator,
                                                  const Guid &componentId) const
{
    return getById_impl<MeshCollider>(mIdState.mMeshColliderIdToGlobalIndex, allocator, componentId);
}

template <> Mesh *World::getAssetById_impl<Mesh>(const PoolAllocator<Mesh> *allocator, const Guid &assetId) const
{
    return getById_impl<Mesh>(mIdState.mMeshIdToGlobalIndex, allocator, assetId);
}

template <>
Material *World::getAssetById_impl<Material>(const PoolAllocator<Material> *allocator, const Guid &assetId) const
{
    return getById_impl<Material>(mIdState.mMaterialIdToGlobalIndex, allocator, assetId);
}

template <> Shader *World::getAssetById_impl<Shader>(const PoolAllocator<Shader> *allocator, const Guid &assetId) const
{
    return getById_impl<Shader>(mIdState.mShaderIdToGlobalIndex, allocator, assetId);
}

template <>
Texture2D *World::getAssetById_impl<Texture2D>(const PoolAllocator<Texture2D> *allocator, const Guid &assetId) const
{
    return getById_impl<Texture2D>(mIdState.mTexture2DIdToGlobalIndex, allocator, assetId);
}

template <>
Texture3D *World::getAssetById_impl<Texture3D>(const PoolAllocator<Texture3D> *allocator, const Guid &assetId) const
{
    return getById_impl<Texture3D>(mIdState.mTexture3DIdToGlobalIndex, allocator, assetId);
}

template <>
Cubemap *World::getAssetById_impl<Cubemap>(const PoolAllocator<Cubemap> *allocator, const Guid &assetId) const
{
    return getById_impl<Cubemap>(mIdState.mCubemapIdToGlobalIndex, allocator, assetId);
}

template <>
RenderTexture *World::getAssetById_impl<RenderTexture>(const PoolAllocator<RenderTexture> *allocator,
                                                const Guid &assetId) const
{
    return getById_impl<RenderTexture>(mIdState.mRenderTextureIdToGlobalIndex, allocator, assetId);
}

template <> Font *World::getAssetById_impl<Font>(const PoolAllocator<Font> *allocator, const Guid &assetId) const
{
    return getById_impl<Font>(mIdState.mFontIdToGlobalIndex, allocator, assetId);
}

template <> Sprite *World::getAssetById_impl<Sprite>(const PoolAllocator<Sprite> *allocator, const Guid &assetId) const
{
    return getById_impl<Sprite>(mIdState.mSpriteIdToGlobalIndex, allocator, assetId);
}

template <>
RenderSystem *World::getSystemById_impl<RenderSystem>(const PoolAllocator<RenderSystem> *allocator,
                                                      const Guid &assetId) const
{
    return getById_impl<RenderSystem>(mIdState.mRenderSystemIdToGlobalIndex, allocator, assetId);
}

template <>
PhysicsSystem *World::getSystemById_impl<PhysicsSystem>(const PoolAllocator<PhysicsSystem> *allocator,
                                                 const Guid &assetId) const
{
    return getById_impl<PhysicsSystem>(mIdState.mPhysicsSystemIdToGlobalIndex, allocator, assetId);
}

template <>
CleanUpSystem *World::getSystemById_impl<CleanUpSystem>(const PoolAllocator<CleanUpSystem> *allocator,
                                                 const Guid &assetId) const
{
    return getById_impl<CleanUpSystem>(mIdState.mCleanupSystemIdToGlobalIndex, allocator, assetId);
}

template <>
DebugSystem *World::getSystemById_impl<DebugSystem>(const PoolAllocator<DebugSystem> *allocator,
                                                    const Guid &assetId) const
{
    return getById_impl<DebugSystem>(mIdState.mDebugSystemIdToGlobalIndex, allocator, assetId);
}

template <>
GizmoSystem *World::getSystemById_impl<GizmoSystem>(const PoolAllocator<GizmoSystem> *allocator,
                                                    const Guid &assetId) const
{
    return getById_impl<GizmoSystem>(mIdState.mGizmoSystemIdToGlobalIndex, allocator, assetId);
}

template <>
FreeLookCameraSystem *World::getSystemById_impl<FreeLookCameraSystem>(
    const PoolAllocator<FreeLookCameraSystem> *allocator,
                                                               const Guid &assetId) const
{
    return getById_impl<FreeLookCameraSystem>(mIdState.mFreeLookCameraSystemIdToGlobalIndex, allocator, assetId);
}

template <> void World::addIdToGlobalIndexMap_impl<Scene>(const Guid &id, int index, int type)
{
    mIdState.mSceneIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Entity>(const Guid &id, int index, int type)
{
    mIdState.mEntityIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Transform>(const Guid &id, int index, int type)
{
    mIdState.mTransformIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<MeshRenderer>(const Guid &id, int index, int type)
{
    mIdState.mMeshRendererIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<SpriteRenderer>(const Guid &id, int index, int type)
{
    mIdState.mSpriteRendererIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<LineRenderer>(const Guid &id, int index, int type)
{
    mIdState.mLineRendererIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Rigidbody>(const Guid &id, int index, int type)
{
    mIdState.mRigidbodyIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Camera>(const Guid &id, int index, int type)
{
    mIdState.mCameraIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Light>(const Guid &id, int index, int type)
{
    mIdState.mLightIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<SphereCollider>(const Guid &id, int index, int type)
{
    mIdState.mSphereColliderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<BoxCollider>(const Guid &id, int index, int type)
{
    mIdState.mBoxColliderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<CapsuleCollider>(const Guid &id, int index, int type)
{
    mIdState.mCapsuleColliderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<MeshCollider>(const Guid &id, int index, int type)
{
    mIdState.mMeshColliderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Mesh>(const Guid &id, int index, int type)
{
    mIdState.mMeshIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Material>(const Guid &id, int index, int type)
{
    mIdState.mMaterialIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Shader>(const Guid &id, int index, int type)
{
    mIdState.mShaderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Texture2D>(const Guid &id, int index, int type)
{
    mIdState.mTexture2DIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Texture3D>(const Guid &id, int index, int type)
{
    mIdState.mTexture3DIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Cubemap>(const Guid &id, int index, int type)
{
    mIdState.mCubemapIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<RenderTexture>(const Guid &id, int index, int type)
{
    mIdState.mRenderTextureIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Font>(const Guid &id, int index, int type)
{
    mIdState.mFontIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Sprite>(const Guid &id, int index, int type)
{
    mIdState.mSpriteIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<RenderSystem>(const Guid &id, int index, int type)
{
    mIdState.mRenderSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<PhysicsSystem>(const Guid &id, int index, int type)
{
    mIdState.mPhysicsSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<CleanUpSystem>(const Guid &id, int index, int type)
{
    mIdState.mCleanupSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<DebugSystem>(const Guid &id, int index, int type)
{
    mIdState.mDebugSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<GizmoSystem>(const Guid &id, int index, int type)
{
    mIdState.mGizmoSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<FreeLookCameraSystem>(const Guid &id, int index, int type)
{
    mIdState.mFreeLookCameraSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}