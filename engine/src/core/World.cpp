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

                        std::cout << "relative data path: " << relativeDataPath.string() << std::endl;

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
        generateSourcePaths(filePath, in.begin()->second);

        Asset *asset = loadAssetFromYAML_impl(in.begin()->second);
        if (asset != nullptr)
        {
            //mIdState.mAssetIdToFilepath[asset->getId()] = filePath;
            //mIdState.mAssetFilepathToId[filePath] = asset->getId();
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

    Scene *scene = loadSceneFromYAML_impl(in);
    if (scene != nullptr)
    {
        //mIdState.mSceneIdToFilepath[scene->getId()] = filePath;
        //mIdState.mSceneFilepathToId[filePath] = scene->getId();
    }

    return scene;
}

bool World::writeAssetToYAML(const std::string &filePath, Id assetId) const
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

bool World::writeSceneToYAML(const std::string &filePath, Id sceneId) const
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

            std::vector<std::pair<Id, int>> temp = entity->getComponentsOnEntity();
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

Asset *World::loadAssetFromYAML_impl(const YAML::Node &in)
{
    int type = YAML::getValue<int>(in, "type");
    Guid guid = YAML::getValue<Guid>(in, "id");

    if (PhysicsEngine::isAsset(type) && guid.isValid())
    {
        if (type == AssetType<Shader>::type)
        {
            std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
            if (it == mIdState.mGuidToId.end())
            {
                int index = (int)mAllocators.mShaderAllocator.getCount();
                Shader *shader = mAllocators.mShaderAllocator.construct(this);

                if (shader != nullptr)
                {
                    mIdState.mGuidToId[shader->getId()] = index;
                    mIdState.mIdToGlobalIndex[shader->getId()] = index;
                    mIdState.mIdToType[shader->getId()] = ComponentType<Shader>::type;

                    mIdState.mGuidToId[guid] = shader->getId();
                    mIdState.mGuidToId[shader->getId()] = guid;
                }
            }       
        }
        else if (type == AssetType<Texture2D>::type)
        {
            std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
            if (it == mIdState.mGuidToId.end())
            {
                int index = (int)mAllocators.mTexture2DAllocator.getCount();
                Texture2D *texture = mAllocators.mTexture2DAllocator.construct(this);

                if (texture != nullptr)
                {
                    mIdState.mGuidToId[texture->getId()] = index;
                    mIdState.mIdToGlobalIndex[texture->getId()] = index;
                    mIdState.mIdToType[texture->getId()] = ComponentType<Texture2D>::type;

                    mIdState.mGuidToId[guid] = texture->getId();
                    mIdState.mGuidToId[texture->getId()] = guid;
                }
            }       
        }
        else if (type == AssetType<Texture3D>::type)
        {
            std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
            if (it == mIdState.mGuidToId.end())
            {
                int index = (int)mAllocators.mTexture3DAllocator.getCount();
                Texture3D *texture = mAllocators.mTexture3DAllocator.construct(this);

                if (texture != nullptr)
                {
                    mIdState.mGuidToId[texture->getId()] = index;
                    mIdState.mIdToGlobalIndex[texture->getId()] = index;
                    mIdState.mIdToType[texture->getId()] = ComponentType<Texture3D>::type;

                    mIdState.mGuidToId[guid] = texture->getId();
                    mIdState.mGuidToId[texture->getId()] = guid;
                }
            }       
        }
        else if (type == AssetType<Cubemap>::type)
        {
            std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
            if (it == mIdState.mGuidToId.end())
            {
                int index = (int)mAllocators.mCubemapAllocator.getCount();
                Cubemap *cubemap = mAllocators.mCubemapAllocator.construct(this);

                if (texture != nullptr)
                {
                    mIdState.mGuidToId[cubemap->getId()] = index;
                    mIdState.mIdToGlobalIndex[cubemap->getId()] = index;
                    mIdState.mIdToType[cubemap->getId()] = ComponentType<Cubemap>::type;

                    mIdState.mGuidToId[guid] = cubemap->getId();
                    mIdState.mGuidToId[cubemap->getId()] = guid;
                }
            }       
        }
        else if (type == AssetType<RenderTexture>::type)
        {
            std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
            if (it == mIdState.mGuidToId.end())
            {
                int index = (int)mAllocators.mRenderTextureAllocator.getCount();
                RenderTexture *renderTexture = mAllocators.mRenderTextureAllocator.construct(this);

                if (renderTexture != nullptr)
                {
                    mIdState.mGuidToId[renderTexture->getId()] = index;
                    mIdState.mIdToGlobalIndex[renderTexture->getId()] = index;
                    mIdState.mIdToType[renderTexture->getId()] = ComponentType<RenderTexture>::type;

                    mIdState.mGuidToId[guid] = renderTexture->getId();
                    mIdState.mGuidToId[renderTexture->getId()] = guid;
                }
            }      
        }
        else if (type == AssetType<Material>::type)
        {
            std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
            if (it == mIdState.mGuidToId.end())
            {
                int index = (int)mAllocators.mMaterialAllocator.getCount();
                Material *material = mAllocators.mMaterialAllocator.construct(this);

                if (material != nullptr)
                {
                    mIdState.mGuidToId[material->getId()] = index;
                    mIdState.mIdToGlobalIndex[material->getId()] = index;
                    mIdState.mIdToType[material->getId()] = ComponentType<Material>::type;

                    mIdState.mGuidToId[guid] = material->getId();
                    mIdState.mGuidToId[material->getId()] = guid;
                }
            }      
        }
        else if (type == AssetType<Mesh>::type)
        {
            std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
            if (it == mIdState.mGuidToId.end())
            {
                int index = (int)mAllocators.mMeshAllocator.getCount();
                Mesh *mesh = mAllocators.mMeshAllocator.construct(this);

                if (mesh != nullptr)
                {
                    mIdState.mGuidToId[mesh->getId()] = index;
                    mIdState.mIdToGlobalIndex[mesh->getId()] = index;
                    mIdState.mIdToType[mesh->getId()] = ComponentType<Mesh>::type;

                    mIdState.mGuidToId[guid] = mesh->getId();
                    mIdState.mGuidToId[mesh->getId()] = guid;
                }
            }      
        }
        else if (type == AssetType<Font>::type)
        {
            std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
            if (it == mIdState.mGuidToId.end())
            {
                int index = (int)mAllocators.mFontAllocator.getCount();
                Font *font = mAllocators.mFontAllocator.construct(this);

                if (font != nullptr)
                {
                    mIdState.mGuidToId[font->getId()] = index;
                    mIdState.mIdToGlobalIndex[font->getId()] = index;
                    mIdState.mIdToType[font->getId()] = ComponentType<Font>::type;

                    mIdState.mGuidToId[guid] = font->getId();
                    mIdState.mGuidToId[font->getId()] = guid;
                }
            }      
        }
        else if (type == AssetType<Sprite>::type)
        {
            std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
            if (it == mIdState.mGuidToId.end())
            {
                int index = (int)mAllocators.mSpriteAllocator.getCount();
                Sprite *sprite = mAllocators.mSpriteAllocator.construct(this);

                if (sprite != nullptr)
                {
                    mIdState.mGuidToId[sprite->getId()] = index;
                    mIdState.mIdToGlobalIndex[sprite->getId()] = index;
                    mIdState.mIdToType[sprite->getId()] = ComponentType<Sprite>::type;

                    mIdState.mGuidToId[guid] = sprite->getId();
                    mIdState.mGuidToId[sprite->getId()] = guid;
                }
            }      
        }
        











        if (type == AssetType<Shader>::type)
        {
            if (it != mIdState.mGuidToId.end())
            {
                Transform *transform = getComponentById<Transform>(it->second);

                if (transform != nullptr)
                {
                    transform->deserialize(in);
                }
            }

        }
        else if (type == AssetType<Texture2D>::type)
        {
            if (it != mIdState.mGuidToId.end())
            {
                Transform *transform = getComponentById<Transform>(it->second);

                if (transform != nullptr)
                {
                    transform->deserialize(in);
                }
            }

        }
        else if (type == AssetType<Texture3D>::type)
        {
            if (it != mIdState.mGuidToId.end())
            {
                Transform *transform = getComponentById<Transform>(it->second);

                if (transform != nullptr)
                {
                    transform->deserialize(in);
                }
            }

        }
        else if (type == AssetType<Cubemap>::type)
        {
            if (it != mIdState.mGuidToId.end())
            {
                Transform *transform = getComponentById<Transform>(it->second);

                if (transform != nullptr)
                {
                    transform->deserialize(in);
                }
            }

        }
        else if (type == AssetType<RenderTexture>::type)
        {
            if (it != mIdState.mGuidToId.end())
            {
                Transform *transform = getComponentById<Transform>(it->second);

                if (transform != nullptr)
                {
                    transform->deserialize(in);
                }
            }
        }
        else if (type == AssetType<Material>::type)
        {
            if (it != mIdState.mGuidToId.end())
            {
                Transform *transform = getComponentById<Transform>(it->second);

                if (transform != nullptr)
                {
                    transform->deserialize(in);
                }
            }
        }
        else if (type == AssetType<Mesh>::type)
        {
            if (it != mIdState.mGuidToId.end())
            {
                Transform *transform = getComponentById<Transform>(it->second);

                if (transform != nullptr)
                {
                    transform->deserialize(in);
                }
            }
        }
        else if (type == AssetType<Font>::type)
        {
            if (it != mIdState.mGuidToId.end())
            {
                Transform *transform = getComponentById<Transform>(it->second);

                if (transform != nullptr)
                {
                    transform->deserialize(in);
                }
            }
        }
        else if (type == AssetType<Sprite>::type)
        {
            if (it != mIdState.mGuidToId.end())
            {
                Transform *transform = getComponentById<Transform>(it->second);

                if (transform != nullptr)
                {
                    transform->deserialize(in);
                }
            }
        }



        
        /*if (Asset::isInternal(type))
        {
            return PhysicsEngine::loadInternalAsset(*this, mAllocators, mIdState, in, guid, type);
        }
        else
        {
            return PhysicsEngine::loadAsset(*this, mAllocators, mIdState, in, guid, type);
        }*/
    }

    return nullptr;
}

Scene *World::loadSceneFromYAML_impl(const YAML::Node &in)
{
    int type = YAML::getValue<int>(in, "type");
    Guid guid = YAML::getValue<Guid>(in, "id");

    if (PhysicsEngine::isScene(type) && guid.isValid())
    {
        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
        if (it == mIdState.mGuidToId.end())
        {
            int index = (int)mAllocators.mSceneAllocator.getCount();
            Scene *scene = mAllocators.mSceneAllocator.construct(this);

            if (scene != nullptr)
            {
                mIdState.mSceneIdToGlobalIndex[scene->getId()] = index;
                mIdState.mIdToGlobalIndex[scene->getId()] = index;
                mIdState.mIdToType[scene->getId()] = SceneType<Scene>::type;

                mIdState.mGuidToId[guid] = scene->getId();
                mIdState.mGuidToId[scene->getId()] = guid;
            }
        }

        for (YAML::const_iterator it = in.begin(); it != in.end(); ++it)
        {
            if (it->first.IsScalar() && it->second.IsMap())
            {
                int type = YAML::getValue<int>(in, "type");
                Guid guid = YAML::getValue<Guid>(in, "id");
                HideFlag hide = YAML::getValue<HideFlag>(in, "hide");

                if (hide == HideFlag::DontSave){ continue; }

                if (PhysicsEngine::isEntity(type))
                {
                    std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                    if (it == mIdState.mGuidToId.end())
                    {
                        int index = (int)mAllocators.mEntityAllocator.getCount();
                        Entity *entity = mAllocators.mEntityAllocator.construct(this);

                        if (entity != nullptr)
                        {
                            mIdState.mGuidToId[entity->getId()] = index;
                            mIdState.mIdToGlobalIndex[entity->getId()] = index;
                            mIdState.mIdToType[entity->getId()] = EntityType<Entity>::type;

                            mIdState.mGuidToId[guid] = entity->getId();
                            mIdState.mGuidToId[entity->getId()] = guid;
                        }
                    }
                }
                else if (PhysicsEngine::isComponent(type))
                {
                    if (type == ComponentType<Transform>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mTransformAllocator.getCount();
                            Transform *transform = mAllocators.mTransformAllocator.construct(this);

                            if (transform != nullptr)
                            {
                                mIdState.mGuidToId[transform->getId()] = index;
                                mIdState.mIdToGlobalIndex[transform->getId()] = index;
                                mIdState.mIdToType[transform->getId()] = ComponentType<Transform>::type;

                                mIdState.mGuidToId[guid] = transform->getId();
                                mIdState.mGuidToId[transform->getId()] = guid;
                            }
                        }       
                    }
                    else if (type == ComponentType<Rigidbody>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mRigidbodyAllocator.getCount();
                            Rigidbody *rigidbody = mAllocators.mRigidbodyAllocator.construct(this);

                            if (rigidbody != nullptr)
                            {
                                mIdState.mGuidToId[rigidbody->getId()] = index;
                                mIdState.mIdToGlobalIndex[rigidbody->getId()] = index;
                                mIdState.mIdToType[rigidbody->getId()] = ComponentType<Rigidbody>::type;

                                mIdState.mGuidToId[guid] = rigidbody->getId();
                                mIdState.mGuidToId[rigidbody->getId()] = guid;
                            }
                        }       
                    }
                    else if (type == ComponentType<Camera>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mCameraAllocator.getCount();
                            Camera *camera = mAllocators.mCameraAllocator.construct(this);

                            if (camera != nullptr)
                            {
                                mIdState.mGuidToId[camera->getId()] = index;
                                mIdState.mIdToGlobalIndex[camera->getId()] = index;
                                mIdState.mIdToType[camera->getId()] = ComponentType<Camera>::type;

                                mIdState.mGuidToId[guid] = camera->getId();
                                mIdState.mGuidToId[camera->getId()] = guid;
                            }
                        }       
                    }
                    else if (type == ComponentType<MeshRenderer>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mMeshRendererAllocator.getCount();
                            MeshRenderer *meshRenderer = mAllocators.mMeshRendererAllocator.construct(this);

                            if (meshRenderer != nullptr)
                            {
                                mIdState.mGuidToId[meshRenderer->getId()] = index;
                                mIdState.mIdToGlobalIndex[meshRenderer->getId()] = index;
                                mIdState.mIdToType[meshRenderer->getId()] = ComponentType<MeshRenderer>::type;

                                mIdState.mGuidToId[guid] = meshRenderer->getId();
                                mIdState.mGuidToId[meshRenderer->getId()] = guid;
                            }
                        }
                    }
                    else if (type == ComponentType<SpriteRenderer>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mSpriteRendererAllocator.getCount();
                            SpriteRenderer *spriteRenderer = mAllocators.mSpriteRendererAllocator.construct(this);

                            if (spriteRenderer != nullptr)
                            {
                                mIdState.mGuidToId[spriteRenderer->getId()] = index;
                                mIdState.mIdToGlobalIndex[spriteRenderer->getId()] = index;
                                mIdState.mIdToType[spriteRenderer->getId()] = ComponentType<SpriteRenderer>::type;

                                mIdState.mGuidToId[guid] = spriteRenderer->getId();
                                mIdState.mGuidToId[spriteRenderer->getId()] = guid;
                            }
                        }                               
                    }
                    else if (type == ComponentType<LineRenderer>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mLineRendererAllocator.getCount();
                            LineRenderer *lineRenderer = mAllocators.mLineRendererAllocator.construct(this);

                            if (lineRenderer != nullptr)
                            {
                                mIdState.mGuidToId[lineRenderer->getId()] = index;
                                mIdState.mIdToGlobalIndex[lineRenderer->getId()] = index;
                                mIdState.mIdToType[lineRenderer->getId()] = ComponentType<LineRenderer>::type;

                                mIdState.mGuidToId[guid] = lineRenderer->getId();
                                mIdState.mGuidToId[lineRenderer->getId()] = guid;
                            }
                        }         
                    }
                    else if (type == ComponentType<Light>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mLightAllocator.getCount();
                            Light *light = mAllocators.mLightAllocator.construct(this);

                            if (light != nullptr)
                            {
                                mIdState.mGuidToId[light->getId()] = index;
                                mIdState.mIdToGlobalIndex[light->getId()] = index;
                                mIdState.mIdToType[light->getId()] = ComponentType<Light>::type;

                                mIdState.mGuidToId[guid] = light->getId();
                                mIdState.mGuidToId[light->getId()] = guid;
                            }
                        }        
                    }
                    else if (type == ComponentType<BoxCollider>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mBoxColliderAllocator.getCount();
                            BoxCollider *box = mAllocators.mBoxColliderAllocator.construct(this);

                            if (box != nullptr)
                            {
                                mIdState.mGuidToId[box->getId()] = index;
                                mIdState.mIdToGlobalIndex[box->getId()] = index;
                                mIdState.mIdToType[box->getId()] = ComponentType<BoxCollider>::type;

                                mIdState.mGuidToId[guid] = box->getId();
                                mIdState.mGuidToId[box->getId()] = guid;
                            }
                        }
                    }
                    else if (type == ComponentType<SphereCollider>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mSphereColliderAllocator.getCount();
                            SphereCollider *sphere = mAllocators.mSphereColliderAllocator.construct(this);

                            if (sphere != nullptr)
                            {
                                mIdState.mGuidToId[sphere->getId()] = index;
                                mIdState.mIdToGlobalIndex[sphere->getId()] = index;
                                mIdState.mIdToType[sphere->getId()] = ComponentType<SphereCollider>::type;

                                mIdState.mGuidToId[guid] = sphere->getId();
                                mIdState.mGuidToId[sphere->getId()] = guid;
                            }
                        }
                    }
                    else if (type == ComponentType<MeshCollider>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mMeshColliderAllocator.getCount();
                            MeshCollider *mesh = mAllocators.mMeshColliderAllocator.construct(this);

                            if (mesh != nullptr)
                            {
                                mIdState.mGuidToId[mesh->getId()] = index;
                                mIdState.mIdToGlobalIndex[mesh->getId()] = index;
                                mIdState.mIdToType[mesh->getId()] = ComponentType<MeshCollider>::type;

                                mIdState.mGuidToId[guid] = mesh->getId();
                                mIdState.mGuidToId[mesh->getId()] = guid;
                            }
                        }
                    }
                    else if (type == ComponentType<CapsuleCollider>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mCapsuleColliderAllocator.getCount();
                            CapsuleCollider *capsule = mAllocators.mCapsuleColliderAllocator.construct(this);

                            if (capsule != nullptr)
                            {
                                mIdState.mGuidToId[capsule->getId()] = index;
                                mIdState.mIdToGlobalIndex[capsule->getId()] = index;
                                mIdState.mIdToType[capsule->getId()] = ComponentType<CapsuleCollider>::type;

                                mIdState.mGuidToId[guid] = capsule->getId();
                                mIdState.mGuidToId[capsule->getId()] = guid;
                            }
                        }
                    }
                    else if (type == ComponentType<Terrain>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mTerrainAllocator.getCount();
                            Terrain *terrain = mAllocators.mTerrainAllocator.construct(this);

                            if (terrain != nullptr)
                            {
                                mIdState.mGuidToId[terrain->getId()] = index;
                                mIdState.mIdToGlobalIndex[terrain->getId()] = index;
                                mIdState.mIdToType[terrain->getId()] = ComponentType<Terrain>::type;

                                mIdState.mGuidToId[guid] = terrain->getId();
                                mIdState.mGuidToId[terrain->getId()] = guid;
                            }
                        }
                    }
                }
                else if (PhysicsEngine::isSystem(type))
                {
                    if (type == SystemType<RenderSystem>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mRenderSystemAllocator.getCount();
                            RenderSystem *renderSystem = mAllocators.mRenderSystemAllocator.construct(this);

                            if (renderSystem != nullptr)
                            {
                                mIdState.mGuidToId[renderSystem->getId()] = index;
                                mIdState.mIdToGlobalIndex[renderSystem->getId()] = index;
                                mIdState.mIdToType[renderSystem->getId()] = ComponentType<RenderSystem>::type;

                                mIdState.mGuidToId[guid] = renderSystem->getId();
                                mIdState.mGuidToId[renderSystem->getId()] = guid;
                            }
                        }
                    }
                    else if (type == SystemType<PhysicsSystem>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mPhysicsSystemAllocator.getCount();
                            PhysicsSystem *physicsSystem = mAllocators.mPhysicsSystemAllocator.construct(this);

                            if (physicsSystem != nullptr)
                            {
                                mIdState.mGuidToId[physicsSystem->getId()] = index;
                                mIdState.mIdToGlobalIndex[physicsSystem->getId()] = index;
                                mIdState.mIdToType[physicsSystem->getId()] = ComponentType<PhysicsSystem>::type;

                                mIdState.mGuidToId[guid] = physicsSystem->getId();
                                mIdState.mGuidToId[physicsSystem->getId()] = guid;
                            }
                        }
                    }
                    else if (type == SystemType<FreeLookCameraSystem>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mFreeLookCameraSystemAllocator.getCount();
                            FreeLookCameraSystem *cameraSystem =
                                mAllocators.mFreeLookCameraSystemAllocator.construct(this);

                            if (cameraSystem != nullptr)
                            {
                                mIdState.mGuidToId[cameraSystem->getId()] = index;
                                mIdState.mIdToGlobalIndex[cameraSystem->getId()] = index;
                                mIdState.mIdToType[cameraSystem->getId()] = ComponentType<FreeLookCameraSystem>::type;

                                mIdState.mGuidToId[guid] = cameraSystem->getId();
                                mIdState.mGuidToId[cameraSystem->getId()] = guid;
                            }
                        }
                    }
                    else if (type == SystemType<CleanUpSystem>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mCleanupSystemAllocator.getCount();
                            CleanUpSystem *cleanupSystem = mAllocators.mCleanupSystemAllocator.construct(this);

                            if (cleanupSystem != nullptr)
                            {
                                mIdState.mGuidToId[cleanupSystem->getId()] = index;
                                mIdState.mIdToGlobalIndex[cleanupSystem->getId()] = index;
                                mIdState.mIdToType[cleanupSystem->getId()] = ComponentType<CleanUpSystem>::type;

                                mIdState.mGuidToId[guid] = cleanupSystem->getId();
                                mIdState.mGuidToId[cleanupSystem->getId()] = guid;
                            }
                        }
                    }
                    else if (type == SystemType<DebugSystem>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mDebugSystemAllocator.getCount();
                            DebugSystem *debugSystem = mAllocators.mDebugSystemAllocator.construct(this);

                            if (debugSystem != nullptr)
                            {
                                mIdState.mGuidToId[debugSystem->getId()] = index;
                                mIdState.mIdToGlobalIndex[debugSystem->getId()] = index;
                                mIdState.mIdToType[debugSystem->getId()] = ComponentType<DebugSystem>::type;

                                mIdState.mGuidToId[guid] = debugSystem->getId();
                                mIdState.mGuidToId[debugSystem->getId()] = guid;
                            }
                        }
                    }
                    else if (type == SystemType<GizmoSystem>::type)
                    {
                        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
                        if (it == mIdState.mGuidToId.end())
                        {
                            int index = (int)mAllocators.mGizmoSystemAllocator.getCount();
                            GizmoSystem *gizmoSystem = mAllocators.mGizmoSystemAllocator.construct(this);

                            if (gizmoSystem != nullptr)
                            {
                                mIdState.mGuidToId[gizmoSystem->getId()] = index;
                                mIdState.mIdToGlobalIndex[gizmoSystem->getId()] = index;
                                mIdState.mIdToType[gizmoSystem->getId()] = ComponentType<GizmoSystem>::type;

                                mIdState.mGuidToId[guid] = gizmoSystem->getId();
                                mIdState.mGuidToId[gizmoSystem->getId()] = guid;
                            }
                        }
                    }
                }
            }
        }

        std::unordered_map<Id, int>::iterator it = mIdState.mGuidToId.find(guid);
        if (it != mIdState.mGuidToId.end())
        {
            Scene *scene = getSceneById(it->second);
            
            if (scene != nullptr)
            {
                scene->deserialize(in);
            }
        }

        for (YAML::const_iterator it = in.begin(); it != in.end(); ++it)
        {
            if (it->first.IsScalar() && it->second.IsMap())
            {
                int type = YAML::getValue<int>(in, "type");
                Guid guid = YAML::getValue<Guid>(in, "id");
                HideFlag hide = YAML::getValue<HideFlag>(in, "hide");

                if (hide == HideFlag::DontSave)
                {
                    continue;
                }

                if (PhysicsEngine::isEntity(type))
                {
                    if (it != mIdState.mGuidToId.end())
                    {
                        Entity *entity = getEntityById(it->second);

                        if (entity != nullptr)
                        {
                            entity->deserialize(in);
                        }
                    }
                }
                else if (PhysicsEngine::isComponent(type))
                {
                    if (type == ComponentType<Transform>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            Transform *transform = getComponentById<Transform>(it->second);

                            if (transform != nullptr)
                            {
                                transform->deserialize(in);
                            }
                        }
                    }
                    else if (type == ComponentType<Rigidbody>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            Rigidbody *rigidbody = getComponentById<Rigidbody>(it->second);

                            if (rigidbody != nullptr)
                            {
                                rigidbody->deserialize(in);
                            }
                        }
                    }
                    else if (type == ComponentType<Camera>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            Camera *camera = getComponentById<Camera>(it->second);

                            if (camera != nullptr)
                            {
                                camera->deserialize(in);
                            }
                        }
                    }
                    else if (type == ComponentType<MeshRenderer>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            MeshRenderer *meshRenderer = getComponentById<MeshRenderer>(it->second);

                            if (meshRenderer != nullptr)
                            {
                                meshRenderer->deserialize(in);
                            }
                        }
                    }
                    else if (type == ComponentType<SpriteRenderer>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            SpriteRenderer *spriteRenderer = getComponentById<SpriteRenderer>(it->second);

                            if (spriteRenderer != nullptr)
                            {
                                spriteRenderer->deserialize(in);
                            }
                        }
                    }
                    else if (type == ComponentType<LineRenderer>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            LineRenderer *lineRenderer = getComponentById<LineRenderer>(it->second);

                            if (lineRenderer != nullptr)
                            {
                                lineRenderer->deserialize(in);
                            }
                        }
                    }
                    else if (type == ComponentType<Light>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            Light *light = getComponentById<Light>(it->second);

                            if (light != nullptr)
                            {
                                light->deserialize(in);
                            }
                        }
                    }
                    else if (type == ComponentType<BoxCollider>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            BoxCollider *box = getComponentById<BoxCollider>(it->second);

                            if (box != nullptr)
                            {
                                box->deserialize(in);
                            }
                        }
                    }
                    else if (type == ComponentType<SphereCollider>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            SphereCollider *sphere = getComponentById<SphereCollider>(it->second);

                            if (sphere != nullptr)
                            {
                                sphere->deserialize(in);
                            }
                        }
                    }
                    else if (type == ComponentType<MeshCollider>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            MeshCollider *mesh = getComponentById<MeshCollider>(it->second);

                            if (mesh != nullptr)
                            {
                                mesh->deserialize(in);
                            }
                        }
                    }
                    else if (type == ComponentType<CapsuleCollider>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            CapsuleCollider *capsule = getComponentById<CapsuleCollider>(it->second);

                            if (capsule != nullptr)
                            {
                                capsule->deserialize(in);
                            }
                        }
                    }
                    else if (type == ComponentType<Terrain>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            Terrain *terrain = getComponentById<Terrain>(it->second);

                            if (terrain != nullptr)
                            {
                                terrain->deserialize(in);
                            }
                        }
                    }
                }
                else if (PhysicsEngine::isSystem(type))
                {
                    if (type == SystemType<RenderSystem>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            RenderSystem *renderSystem = getSystemById<RenderSystem>(it->second);

                            if (renderSystem != nullptr)
                            {
                                renderSystem->deserialize(in);
                            }
                        }
                    }
                    else if (type == SystemType<PhysicsSystem>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            PhysicsSystem *physicsSystem = getSystemById<PhysicsSystem>(it->second);

                            if (physicsSystem != nullptr)
                            {
                                physicsSystem->deserialize(in);
                            }
                        }
                    }
                    else if (type == SystemType<FreeLookCameraSystem>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            FreeLookCameraSystem *cameraSystem = getSystemById<FreeLookCameraSystem>(it->second);

                            if (cameraSystem != nullptr)
                            {
                                cameraSystem->deserialize(in);
                            }
                        }
                    }
                    else if (type == SystemType<CleanUpSystem>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            CleanUpSystem *cleanupSystem = getSystemById<CleanUpSystem>(it->second);

                            if (cleanupSystem != nullptr)
                            {
                                cleanupSystem->deserialize(in);
                            }
                        }
                    }
                    else if (type == SystemType<DebugSystem>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            DebugSystem *debugSystem = getSystemById<DebugSystem>(it->second);

                            if (debugSystem != nullptr)
                            {
                                debugSystem->deserialize(in);
                            }
                        }
                    }
                    else if (type == SystemType<GizmoSystem>::type)
                    {
                        if (it != mIdState.mGuidToId.end())
                        {
                            GizmoSystem *gizmoSystem = getSystemById<GizmoSystem>(it->second);

                            if (gizmoSystem != nullptr)
                            {
                                gizmoSystem->deserialize(in);
                            }
                        }
                    }
                }
            }
        }
    }

    return nullptr;
}

void World::generateSourcePaths(const std::string &filepath, YAML::Node &in)
{
    int type = YAML::getValue<int>(in, "type");

    std::filesystem::path path = filepath;
    path.remove_filename();

    if (PhysicsEngine::isAsset(type))
    {
        switch (type)
        {
        case AssetType<Shader>::type:
        case AssetType<Texture2D>::type:
        case AssetType<Mesh>::type: {
            std::filesystem::path source = YAML::getValue<std::string>(in, "source");
            in["sourceFilepath"] = (path / source).string();
            break;
        }
        }
    }
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
    std::vector<Id> entitiesToDestroy;
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

std::vector<ShaderUniform> World::getCachedMaterialUniforms(Id materialId, Id shaderId)
{
    return mMaterialUniformCache[materialId][shaderId];
}

void World::cacheMaterialUniforms(Id materialId, Id shaderId, const std::vector<ShaderUniform> &uniforms)
{
    assert(materialId != -1);
    assert(shaderId != -1);

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
    assert(mesh != nullptr);
    assert(entity != nullptr);
    
    MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>();
    assert(meshRenderer != nullptr);

    Transform *transform = entity->getComponent<Transform>();
    assert(transform != nullptr);

    entity->setName(mesh->getName());
    
    //transform->mPosition = glm::vec3(0, 0, 0);
    //transform->mRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    //transform->mScale = glm::vec3(1, 1, 1);
    transform->setPosition(glm::vec3(0, 0, 0));
    transform->setRotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    transform->setScale(glm::vec3(1, 1, 1));
    meshRenderer->setMesh(mesh->getId());
    meshRenderer->setMaterial(mPrimitives.mStandardMaterialId);

    return entity;
}

Entity *World::createNonPrimitive(Id meshId)
{
    Mesh *mesh = getAssetById<Mesh>(meshId);
    if (mesh == nullptr)
    {
        return nullptr;
    }

    Entity *entity = createEntity();
    assert(entity != nullptr);

    MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>();
    assert(meshRenderer != nullptr);

    Transform *transform = entity->getComponent<Transform>();
    assert(transform != nullptr);

    entity->setName(mesh->getName());
   
    //transform->mPosition = glm::vec3(0, 0, 0);
    //transform->mRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    //transform->mScale = glm::vec3(1, 1, 1);
    transform->setPosition(glm::vec3(0, 0, 0));
    transform->setRotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    transform->setScale(glm::vec3(1, 1, 1));
    meshRenderer->setMesh(meshId);
    meshRenderer->setMaterial(mPrimitives.mStandardMaterialId);

    return entity;
}

Entity *World::createLight(LightType type)
{
    Entity *entity = createEntity();
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
    entity->addComponent<Camera>();

    return entity;
}

Scene *World::getSceneById(Id sceneId) const
{
    return getById_impl<Scene>(mIdState.mSceneIdToGlobalIndex, &mAllocators.mSceneAllocator, sceneId);
}

Scene *World::getSceneByIndex(size_t index) const
{
    return mAllocators.mSceneAllocator.get(index);
}

Entity *World::getEntityById(Id entityId) const
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

Guid World::getGuidOf(Id id) const
{
    std::unordered_map<Id, Guid>::const_iterator it = mIdState.mIdToGuid.find(id);
    if (it != mIdState.mIdToGuid.end())
    {
        return it->second;
    }

    return Guid::INVALID;
}

Id World::getIdOf(const Guid &guid) const
{
    std::unordered_map<Guid, Id>::const_iterator it = mIdState.mGuidToId.find(guid);
    if (it != mIdState.mGuidToId.end())
    {
        return it->second;
    }

    return -1;
}

int World::getIndexOf(Id id) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mIdToGlobalIndex.find(id);
    if (it != mIdState.mIdToGlobalIndex.end())
    {
        return it->second;
    }

    return -1;
}

int World::getTypeOf(Id id) const
{
    std::unordered_map<Id, int>::const_iterator it = mIdState.mIdToType.find(id);
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

    Scene *scene = mAllocators.mSceneAllocator.construct(this);

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

    Entity *entity = mAllocators.mEntityAllocator.construct(this);

    if (entity != nullptr)
    {
        addIdToGlobalIndexMap_impl<Entity>(entity->getId(), globalIndex, type);

        mIdState.mEntityIdToComponentIds[entity->getId()] = std::vector<std::pair<Id, int>>();

        mIdState.mEntityIdsMarkedCreated.push_back(entity->getId());
    }

    // Add transform (all entitie must have a transform)
    int componentGlobalIndex = (int)mAllocators.mTransformAllocator.getCount();
    int componentType = ComponentType<Transform>::type;

    Transform *component = mAllocators.mTransformAllocator.construct(this);

    assert(component != nullptr);

    component->mEntityId = entity->getId();

    addIdToGlobalIndexMap_impl<Transform>(component->getId(), componentGlobalIndex, componentType);

    mIdState.mEntityIdToComponentIds[entity->getId()].push_back(std::make_pair(component->getId(), componentType));

    mIdState.mComponentIdsMarkedCreated.push_back(std::make_tuple(entity->getId(), component->getId(), componentType));

    return entity;
}

Entity *World::createEntity(const std::string& name)
{
    Entity *entity = createEntity();
    if (entity != nullptr)
    {
        entity->setName(name);
        return entity;
    }

    return nullptr;
}

void World::latentDestroyEntity(Id entityId)
{
    mIdState.mEntityIdsMarkedLatentDestroy.push_back(entityId);

    // add any components found on the entity to the latent destroy component list
    std::unordered_map<Id, std::vector<std::pair<Id, int>>>::const_iterator it =
        mIdState.mEntityIdToComponentIds.find(entityId);

    assert(it != mIdState.mEntityIdToComponentIds.end());

    for (size_t i = 0; i < it->second.size(); i++)
    {
        latentDestroyComponent(entityId, it->second[i].first, it->second[i].second);
    }
}

void World::immediateDestroyEntity(Id entityId)
{
    std::vector<std::pair<Id, int>> componentsOnEntity = mIdState.mEntityIdToComponentIds[entityId];
    for (size_t i = 0; i < componentsOnEntity.size(); i++)
    {
        immediateDestroyComponent(entityId, componentsOnEntity[i].first, componentsOnEntity[i].second);
    }

    assert(mIdState.mEntityIdToComponentIds[entityId].size() == 0);

    mIdState.mEntityIdToComponentIds.erase(entityId);

    destroyInternalEntity(mAllocators, mIdState, entityId, getIndexOf(entityId));
}

void World::latentDestroyComponent(Id entityId, Id componentId, int componentType)
{
    mIdState.mComponentIdsMarkedLatentDestroy.push_back(std::make_tuple(entityId, componentId, componentType));
}

void World::immediateDestroyComponent(Id entityId, Id componentId, int componentType)
{
    // remove from entity component list
    std::vector<std::pair<Id, int>> &componentsOnEntity = mIdState.mEntityIdToComponentIds[entityId];

    std::vector<std::pair<Id, int>>::iterator it = componentsOnEntity.begin();
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

void World::latentDestroyAsset(Id assetId, int assetType)
{
    mIdState.mAssetIdsMarkedLatentDestroy.push_back(std::make_pair(assetId, assetType));
}

void World::immediateDestroyAsset(Id assetId, int assetType)
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

bool World::isMarkedForLatentDestroy(Id id)
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

std::vector<std::pair<Id, int>> World::getComponentsOnEntity(Id entityId) const
{
    std::unordered_map<Id, std::vector<std::pair<Id, int>>>::const_iterator it =
        mIdState.mEntityIdToComponentIds.find(entityId);
    if (it != mIdState.mEntityIdToComponentIds.end())
    {
        return it->second;
    }

    return std::vector<std::pair<Id, int>>();
}

std::vector<Id> World::getEntityIdsMarkedCreated() const
{
    return mIdState.mEntityIdsMarkedCreated;
}

std::vector<Id> World::getEntityIdsMarkedLatentDestroy() const
{
    return mIdState.mEntityIdsMarkedLatentDestroy;
}

std::vector<std::tuple<Id, Id, int>> World::getComponentIdsMarkedCreated() const
{
    return mIdState.mComponentIdsMarkedCreated;
}

std::vector<std::tuple<Id, Id, int>> World::getComponentIdsMarkedLatentDestroy() const
{
    return mIdState.mComponentIdsMarkedLatentDestroy;
}

//std::string World::getAssetFilepath(const Guid &assetId) const
//{
//    std::unordered_map<Guid, std::string>::const_iterator it = mIdState.mAssetIdToFilepath.find(assetId);
//    if (it != mIdState.mAssetIdToFilepath.end())
//    {
//        return it->second;
//    }
//
//    return std::string();
//}
//
//std::string World::getSceneFilepath(const Guid &sceneId) const
//{
//    std::unordered_map<Guid, std::string>::const_iterator it = mIdState.mSceneIdToFilepath.find(sceneId);
//    if (it != mIdState.mSceneIdToFilepath.end())
//    {
//        return it->second;
//    }
//
//    return std::string();
//}
//
//Guid World::getAssetId(const std::string& filepath) const
//{
//    std::unordered_map<std::string, Guid>::const_iterator it = mIdState.mAssetFilepathToId.find(filepath);
//    if (it != mIdState.mAssetFilepathToId.end())
//    {
//        return it->second;
//    }
//
//    return Guid::INVALID;
//}
//
//Guid World::getSceneId(const std::string& filepath) const
//{
//    std::unordered_map<std::string, Guid>::const_iterator it = mIdState.mSceneFilepathToId.find(filepath);
//    if (it != mIdState.mSceneFilepathToId.end())
//    {
//        return it->second;
//    }
//
//    return Guid::INVALID;
//}

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

template <> size_t World::getNumberOfSystems<TerrainSystem>() const
{
    return mAllocators.mTerrainSystemAllocator.getCount();
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

template <> size_t World::getNumberOfComponents<Terrain>() const
{
    return mAllocators.mTerrainAllocator.getCount();
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

template <> TerrainSystem *World::getSystem<TerrainSystem>() const
{
    return getSystem_impl(&mAllocators.mTerrainSystemAllocator);
}

template <> Transform *World::getComponent<Transform>(Id entityId) const
{
    // Transform occurs at same index as its entity since all entities have a transform
    return getComponentByIndex<Transform>(getIndexOf(entityId));
}

template <> MeshRenderer *World::getComponent<MeshRenderer>(Id entityId) const
{
    return getComponent_impl(&mAllocators.mMeshRendererAllocator, entityId);
}

template <> SpriteRenderer *World::getComponent<SpriteRenderer>(Id entityId) const
{
    return getComponent_impl(&mAllocators.mSpriteRendererAllocator, entityId);
}

template <> LineRenderer *World::getComponent<LineRenderer>(Id entityId) const
{
    return getComponent_impl(&mAllocators.mLineRendererAllocator, entityId);
}

template <> Rigidbody *World::getComponent<Rigidbody>(Id entityId) const
{
    return getComponent_impl(&mAllocators.mRigidbodyAllocator, entityId);
}

template <> Camera *World::getComponent<Camera>(Id entityId) const
{
    return getComponent_impl(&mAllocators.mCameraAllocator, entityId);
}

template <> Light *World::getComponent<Light>(Id entityId) const
{
    return getComponent_impl(&mAllocators.mLightAllocator, entityId);
}

template <> SphereCollider *World::getComponent<SphereCollider>(Id entityId) const
{
    return getComponent_impl(&mAllocators.mSphereColliderAllocator, entityId);
}

template <> BoxCollider *World::getComponent<BoxCollider>(Id entityId) const
{
    return getComponent_impl(&mAllocators.mBoxColliderAllocator, entityId);
}

template <> CapsuleCollider *World::getComponent<CapsuleCollider>(Id entityId) const
{
    return getComponent_impl(&mAllocators.mCapsuleColliderAllocator, entityId);
}

template <> MeshCollider *World::getComponent<MeshCollider>(Id entityId) const
{
    return getComponent_impl(&mAllocators.mMeshColliderAllocator, entityId);
}

template <> Terrain *World::getComponent<Terrain>(Id entityId) const
{
    return getComponent_impl(&mAllocators.mTerrainAllocator, entityId);
}

template <> MeshRenderer *World::addComponent<MeshRenderer>(Id entityId)
{
    return addComponent_impl(&mAllocators.mMeshRendererAllocator, entityId);
}

template <> SpriteRenderer *World::addComponent<SpriteRenderer>(Id entityId)
{
    return addComponent_impl(&mAllocators.mSpriteRendererAllocator, entityId);
}

template <> LineRenderer *World::addComponent<LineRenderer>(Id entityId)
{
    return addComponent_impl(&mAllocators.mLineRendererAllocator, entityId);
}

template <> Rigidbody *World::addComponent<Rigidbody>(Id entityId)
{
    return addComponent_impl(&mAllocators.mRigidbodyAllocator, entityId);
}

template <> Camera *World::addComponent<Camera>(Id entityId)
{
    return addComponent_impl(&mAllocators.mCameraAllocator, entityId);
}

template <> Light *World::addComponent<Light>(Id entityId)
{
    return addComponent_impl(&mAllocators.mLightAllocator, entityId);
}

template <> SphereCollider *World::addComponent<SphereCollider>(Id entityId)
{
    return addComponent_impl(&mAllocators.mSphereColliderAllocator, entityId);
}

template <> BoxCollider *World::addComponent<BoxCollider>(Id entityId)
{
    return addComponent_impl(&mAllocators.mBoxColliderAllocator, entityId);
}

template <> CapsuleCollider *World::addComponent<CapsuleCollider>(Id entityId)
{
    return addComponent_impl(&mAllocators.mCapsuleColliderAllocator, entityId);
}

template <> MeshCollider *World::addComponent<MeshCollider>(Id entityId)
{
    return addComponent_impl(&mAllocators.mMeshColliderAllocator, entityId);
}

template <> Terrain *World::addComponent<Terrain>(Id entityId)
{
    return addComponent_impl(&mAllocators.mTerrainAllocator, entityId);
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

template <> TerrainSystem *World::addSystem<TerrainSystem>(size_t order)
{
    return addSystem_impl(&mAllocators.mTerrainSystemAllocator, order);
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

template <> TerrainSystem *World::getSystemByIndex<TerrainSystem>(size_t index) const
{
    return getSystemByIndex_impl(&mAllocators.mTerrainSystemAllocator, index);
}

template <> RenderSystem *World::getSystemById<RenderSystem>(Id systemId) const
{
    return getSystemById_impl(&mAllocators.mRenderSystemAllocator, systemId);
}

template <> PhysicsSystem *World::getSystemById<PhysicsSystem>(Id systemId) const
{
    return getSystemById_impl(&mAllocators.mPhysicsSystemAllocator, systemId);
}

template <> CleanUpSystem *World::getSystemById<CleanUpSystem>(Id systemId) const
{
    return getSystemById_impl(&mAllocators.mCleanupSystemAllocator, systemId);
}

template <> DebugSystem *World::getSystemById<DebugSystem>(Id systemId) const
{
    return getSystemById_impl(&mAllocators.mDebugSystemAllocator, systemId);
}

template <> GizmoSystem *World::getSystemById<GizmoSystem>(Id systemId) const
{
    return getSystemById_impl(&mAllocators.mGizmoSystemAllocator, systemId);
}

template <> FreeLookCameraSystem *World::getSystemById<FreeLookCameraSystem>(Id systemId) const
{
    return getSystemById_impl(&mAllocators.mFreeLookCameraSystemAllocator, systemId);
}

template <> TerrainSystem *World::getSystemById<TerrainSystem>(Id systemId) const
{
    return getSystemById_impl(&mAllocators.mTerrainSystemAllocator, systemId);
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

template <> Mesh *World::getAssetById<Mesh>(Id assetId) const
{
    return getAssetById_impl(&mAllocators.mMeshAllocator, assetId);
}

template <> Material *World::getAssetById<Material>(Id assetId) const
{
    return getAssetById_impl(&mAllocators.mMaterialAllocator, assetId);
}

template <> Shader *World::getAssetById<Shader>(Id assetId) const
{
    return getAssetById_impl(&mAllocators.mShaderAllocator, assetId);
}

template <> Texture2D *World::getAssetById<Texture2D>(Id assetId) const
{
    return getAssetById_impl(&mAllocators.mTexture2DAllocator, assetId);
}

template <> Texture3D *World::getAssetById<Texture3D>(Id assetId) const
{
    return getAssetById_impl(&mAllocators.mTexture3DAllocator, assetId);
}

template <> Cubemap *World::getAssetById<Cubemap>(Id assetId) const
{
    return getAssetById_impl(&mAllocators.mCubemapAllocator, assetId);
}

template <> RenderTexture *World::getAssetById<RenderTexture>(Id assetId) const
{
    return getAssetById_impl(&mAllocators.mRenderTextureAllocator, assetId);
}

template <> Font *World::getAssetById<Font>(Id assetId) const
{
    return getAssetById_impl(&mAllocators.mFontAllocator, assetId);
}

template <> Sprite *World::getAssetById<Sprite>(Id assetId) const
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

template <> Terrain *World::getComponentByIndex<Terrain>(size_t index) const
{
    return getComponentByIndex_impl(&mAllocators.mTerrainAllocator, index);
}

template <> Transform *World::getComponentById<Transform>(Id componentId) const
{
    return getComponentById_impl(&mAllocators.mTransformAllocator, componentId);
}

template <> MeshRenderer *World::getComponentById<MeshRenderer>(Id componentId) const
{
    return getComponentById_impl(&mAllocators.mMeshRendererAllocator, componentId);
}

template <> SpriteRenderer *World::getComponentById<SpriteRenderer>(Id componentId) const
{
    return getComponentById_impl(&mAllocators.mSpriteRendererAllocator, componentId);
}

template <> LineRenderer *World::getComponentById<LineRenderer>(Id componentId) const
{
    return getComponentById_impl(&mAllocators.mLineRendererAllocator, componentId);
}

template <> Rigidbody *World::getComponentById<Rigidbody>(Id componentId) const
{
    return getComponentById_impl(&mAllocators.mRigidbodyAllocator, componentId);
}

template <> Camera *World::getComponentById<Camera>(Id componentId) const
{
    return getComponentById_impl(&mAllocators.mCameraAllocator, componentId);
}

template <> Light *World::getComponentById<Light>(Id componentId) const
{
    return getComponentById_impl(&mAllocators.mLightAllocator, componentId);
}

template <> SphereCollider *World::getComponentById<SphereCollider>(Id componentId) const
{
    return getComponentById_impl(&mAllocators.mSphereColliderAllocator, componentId);
}

template <> BoxCollider *World::getComponentById<BoxCollider>(Id componentId) const
{
    return getComponentById_impl(&mAllocators.mBoxColliderAllocator, componentId);
}

template <> CapsuleCollider *World::getComponentById<CapsuleCollider>(Id componentId) const
{
    return getComponentById_impl(&mAllocators.mCapsuleColliderAllocator, componentId);
}

template <> MeshCollider *World::getComponentById<MeshCollider>(Id componentId) const
{
    return getComponentById_impl(&mAllocators.mMeshColliderAllocator, componentId);
}

template <> Terrain *World::getComponentById<Terrain>(Id componentId) const
{
    return getComponentById_impl(&mAllocators.mTerrainAllocator, componentId);
}

template <> Mesh *World::createAsset<Mesh>()
{
    return createAsset_impl(&mAllocators.mMeshAllocator);
}

template <> Material *World::createAsset<Material>()
{
    return createAsset_impl(&mAllocators.mMaterialAllocator);
}

template <> Shader *World::createAsset<Shader>()
{
    return createAsset_impl(&mAllocators.mShaderAllocator);
}

template <> Texture2D *World::createAsset<Texture2D>()
{
    return createAsset_impl(&mAllocators.mTexture2DAllocator);
}

template <> Texture3D *World::createAsset<Texture3D>()
{
    return createAsset_impl(&mAllocators.mTexture3DAllocator);
}

template <> Cubemap *World::createAsset<Cubemap>()
{
    return createAsset_impl(&mAllocators.mCubemapAllocator);
}

template <> RenderTexture *World::createAsset<RenderTexture>()
{
    return createAsset_impl(&mAllocators.mRenderTextureAllocator);
}

template <> Font *World::createAsset<Font>()
{
    return createAsset_impl(&mAllocators.mFontAllocator);
}

template <> Sprite *World::createAsset<Sprite>()
{
    return createAsset_impl(&mAllocators.mSpriteAllocator);
}

template <>
Transform *World::getComponentById_impl<Transform>(const PoolAllocator<Transform> *allocator,
                                                   Id componentId) const
{
    return getById_impl<Transform>(mIdState.mTransformIdToGlobalIndex, allocator, componentId);
}

template <>
MeshRenderer *World::getComponentById_impl<MeshRenderer>(const PoolAllocator<MeshRenderer> *allocator,
                                                  Id componentId) const
{
    return getById_impl<MeshRenderer>(mIdState.mMeshRendererIdToGlobalIndex, allocator, componentId);
}

template <>
SpriteRenderer *World::getComponentById_impl<SpriteRenderer>(const PoolAllocator<SpriteRenderer> *allocator,
                                                      Id componentId) const
{
    return getById_impl<SpriteRenderer>(mIdState.mSpriteRendererIdToGlobalIndex, allocator, componentId);
}

template <>
LineRenderer *World::getComponentById_impl<LineRenderer>(const PoolAllocator<LineRenderer> *allocator,
                                                  Id componentId) const
{
    return getById_impl<LineRenderer>(mIdState.mLineRendererIdToGlobalIndex, allocator, componentId);
}

template <>
Rigidbody *World::getComponentById_impl<Rigidbody>(const PoolAllocator<Rigidbody> *allocator,
                                                   Id componentId) const
{
    return getById_impl<Rigidbody>(mIdState.mRigidbodyIdToGlobalIndex, allocator, componentId);
}

template <>
Camera *World::getComponentById_impl<Camera>(const PoolAllocator<Camera> *allocator, Id componentId) const
{
    return getById_impl<Camera>(mIdState.mCameraIdToGlobalIndex, allocator, componentId);
}

template <>
Light *World::getComponentById_impl<Light>(const PoolAllocator<Light> *allocator, Id componentId) const
{
    return getById_impl<Light>(mIdState.mLightIdToGlobalIndex, allocator, componentId);
}

template <>
SphereCollider *World::getComponentById_impl<SphereCollider>(const PoolAllocator<SphereCollider> *allocator,
                                                      Id componentId) const
{
    return getById_impl<SphereCollider>(mIdState.mSphereColliderIdToGlobalIndex, allocator, componentId);
}

template <>
BoxCollider *World::getComponentById_impl<BoxCollider>(const PoolAllocator<BoxCollider> *allocator,
                                                Id componentId) const
{
    return getById_impl<BoxCollider>(mIdState.mBoxColliderIdToGlobalIndex, allocator, componentId);
}

template <>
CapsuleCollider *World::getComponentById_impl<CapsuleCollider>(const PoolAllocator<CapsuleCollider> *allocator,
                                                        Id componentId) const
{
    return getById_impl<CapsuleCollider>(mIdState.mCapsuleColliderIdToGlobalIndex, allocator, componentId);
}

template <>
MeshCollider *World::getComponentById_impl<MeshCollider>(const PoolAllocator<MeshCollider> *allocator,
                                                  Id componentId) const
{
    return getById_impl<MeshCollider>(mIdState.mMeshColliderIdToGlobalIndex, allocator, componentId);
}

template <>
Terrain *World::getComponentById_impl<Terrain>(const PoolAllocator<Terrain> *allocator,
                                                         Id componentId) const
{
    return getById_impl<Terrain>(mIdState.mTerrainIdToGlobalIndex, allocator, componentId);
}

template <> Mesh *World::getAssetById_impl<Mesh>(const PoolAllocator<Mesh> *allocator, Id assetId) const
{
    return getById_impl<Mesh>(mIdState.mMeshIdToGlobalIndex, allocator, assetId);
}

template <>
Material *World::getAssetById_impl<Material>(const PoolAllocator<Material> *allocator, Id assetId) const
{
    return getById_impl<Material>(mIdState.mMaterialIdToGlobalIndex, allocator, assetId);
}

template <> Shader *World::getAssetById_impl<Shader>(const PoolAllocator<Shader> *allocator, Id assetId) const
{
    return getById_impl<Shader>(mIdState.mShaderIdToGlobalIndex, allocator, assetId);
}

template <>
Texture2D *World::getAssetById_impl<Texture2D>(const PoolAllocator<Texture2D> *allocator, Id assetId) const
{
    return getById_impl<Texture2D>(mIdState.mTexture2DIdToGlobalIndex, allocator, assetId);
}

template <>
Texture3D *World::getAssetById_impl<Texture3D>(const PoolAllocator<Texture3D> *allocator, Id assetId) const
{
    return getById_impl<Texture3D>(mIdState.mTexture3DIdToGlobalIndex, allocator, assetId);
}

template <>
Cubemap *World::getAssetById_impl<Cubemap>(const PoolAllocator<Cubemap> *allocator, Id assetId) const
{
    return getById_impl<Cubemap>(mIdState.mCubemapIdToGlobalIndex, allocator, assetId);
}

template <>
RenderTexture *World::getAssetById_impl<RenderTexture>(const PoolAllocator<RenderTexture> *allocator,
                                                Id assetId) const
{
    return getById_impl<RenderTexture>(mIdState.mRenderTextureIdToGlobalIndex, allocator, assetId);
}

template <> Font *World::getAssetById_impl<Font>(const PoolAllocator<Font> *allocator, Id assetId) const
{
    return getById_impl<Font>(mIdState.mFontIdToGlobalIndex, allocator, assetId);
}

template <> Sprite *World::getAssetById_impl<Sprite>(const PoolAllocator<Sprite> *allocator, Id assetId) const
{
    return getById_impl<Sprite>(mIdState.mSpriteIdToGlobalIndex, allocator, assetId);
}

template <>
RenderSystem *World::getSystemById_impl<RenderSystem>(const PoolAllocator<RenderSystem> *allocator,
                                                      Id assetId) const
{
    return getById_impl<RenderSystem>(mIdState.mRenderSystemIdToGlobalIndex, allocator, assetId);
}

template <>
PhysicsSystem *World::getSystemById_impl<PhysicsSystem>(const PoolAllocator<PhysicsSystem> *allocator,
                                                 Id assetId) const
{
    return getById_impl<PhysicsSystem>(mIdState.mPhysicsSystemIdToGlobalIndex, allocator, assetId);
}

template <>
CleanUpSystem *World::getSystemById_impl<CleanUpSystem>(const PoolAllocator<CleanUpSystem> *allocator,
                                                 Id assetId) const
{
    return getById_impl<CleanUpSystem>(mIdState.mCleanupSystemIdToGlobalIndex, allocator, assetId);
}

template <>
DebugSystem *World::getSystemById_impl<DebugSystem>(const PoolAllocator<DebugSystem> *allocator,
                                                    Id assetId) const
{
    return getById_impl<DebugSystem>(mIdState.mDebugSystemIdToGlobalIndex, allocator, assetId);
}

template <>
GizmoSystem *World::getSystemById_impl<GizmoSystem>(const PoolAllocator<GizmoSystem> *allocator,
                                                    Id assetId) const
{
    return getById_impl<GizmoSystem>(mIdState.mGizmoSystemIdToGlobalIndex, allocator, assetId);
}

template <>
FreeLookCameraSystem *World::getSystemById_impl<FreeLookCameraSystem>(
    const PoolAllocator<FreeLookCameraSystem> *allocator, Id assetId) const
{
    return getById_impl<FreeLookCameraSystem>(mIdState.mFreeLookCameraSystemIdToGlobalIndex, allocator, assetId);
}

template <>
TerrainSystem *World::getSystemById_impl<TerrainSystem>(
    const PoolAllocator<TerrainSystem> *allocator, Id assetId) const
{
    return getById_impl<TerrainSystem>(mIdState.mTerrainSystemIdToGlobalIndex, allocator, assetId);
}

template <> void World::addIdToGlobalIndexMap_impl<Scene>(Id id, int index, int type)
{
    mIdState.mSceneIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Entity>(Id id, int index, int type)
{
    mIdState.mEntityIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Transform>(Id id, int index, int type)
{
    mIdState.mTransformIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<MeshRenderer>(Id id, int index, int type)
{
    mIdState.mMeshRendererIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<SpriteRenderer>(Id id, int index, int type)
{
    mIdState.mSpriteRendererIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<LineRenderer>(Id id, int index, int type)
{
    mIdState.mLineRendererIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Rigidbody>(Id id, int index, int type)
{
    mIdState.mRigidbodyIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Camera>(Id id, int index, int type)
{
    mIdState.mCameraIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Light>(Id id, int index, int type)
{
    mIdState.mLightIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<SphereCollider>(Id id, int index, int type)
{
    mIdState.mSphereColliderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<BoxCollider>(Id id, int index, int type)
{
    mIdState.mBoxColliderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<CapsuleCollider>(Id id, int index, int type)
{
    mIdState.mCapsuleColliderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<MeshCollider>(Id id, int index, int type)
{
    mIdState.mMeshColliderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Terrain>(Id id, int index, int type)
{
    mIdState.mTerrainIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Mesh>(Id id, int index, int type)
{
    mIdState.mMeshIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Material>(Id id, int index, int type)
{
    mIdState.mMaterialIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Shader>(Id id, int index, int type)
{
    mIdState.mShaderIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Texture2D>(Id id, int index, int type)
{
    mIdState.mTexture2DIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Texture3D>(Id id, int index, int type)
{
    mIdState.mTexture3DIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Cubemap>(Id id, int index, int type)
{
    mIdState.mCubemapIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<RenderTexture>(Id id, int index, int type)
{
    mIdState.mRenderTextureIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Font>(Id id, int index, int type)
{
    mIdState.mFontIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<Sprite>(Id id, int index, int type)
{
    mIdState.mSpriteIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<RenderSystem>(Id id, int index, int type)
{
    mIdState.mRenderSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<PhysicsSystem>(Id id, int index, int type)
{
    mIdState.mPhysicsSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<CleanUpSystem>(Id id, int index, int type)
{
    mIdState.mCleanupSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<DebugSystem>(Id id, int index, int type)
{
    mIdState.mDebugSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<GizmoSystem>(Id id, int index, int type)
{
    mIdState.mGizmoSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<FreeLookCameraSystem>(Id id, int index, int type)
{
    mIdState.mFreeLookCameraSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}

template <> void World::addIdToGlobalIndexMap_impl<TerrainSystem>(Id id, int index, int type)
{
    mIdState.mTerrainSystemIdToGlobalIndex[id] = index;
    mIdState.mIdToGlobalIndex[id] = index;
    mIdState.mIdToType[id] = type;
}