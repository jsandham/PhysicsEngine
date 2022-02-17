#include <algorithm>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <random>
#include <unordered_set>

#define GLM_FORCE_RADIANS

#include "glm/gtc/matrix_transform.hpp"

#include "../../include/core/Intersect.h"
#include "../../include/core/World.h"
#include "../../include/core/Log.h"

#include "../../include/systems/RenderSystem.h"

#include "../../include/graphics/DeferredRenderer.h"
#include "../../include/graphics/ForwardRenderer.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem(World* world) : System(world)
{
}

RenderSystem::RenderSystem(World* world, Guid id) : System(world, id)
{
}

RenderSystem::~RenderSystem()
{
}

void RenderSystem::serialize(YAML::Node &out) const
{
    System::serialize(out);
}

void RenderSystem::deserialize(const YAML::Node &in)
{
    System::deserialize(in);
}

int RenderSystem::getType() const
{
    return PhysicsEngine::RENDERSYSTEM_TYPE;
}

std::string RenderSystem::getObjectName() const
{
    return PhysicsEngine::RENDERSYSTEM_NAME;
}

void RenderSystem::init(World *world)
{
    mWorld = world;

    mForwardRenderer.init(mWorld);
    mDeferredRenderer.init(mWorld);
    mDebugRenderer.init(mWorld);
}

void RenderSystem::update(const Input &input, const Time &time)
{
    registerRenderAssets(mWorld);
    registerCameras(mWorld);
    registerLights(mWorld);

    buildRenderObjectsList(mWorld);
    buildSpriteObjectsList(mWorld);

    for (size_t i = 0; i < mWorld->getNumberOfComponents<Camera>(); i++)
    {
        Camera *camera = mWorld->getComponentByIndex<Camera>(i);

        if (camera->mEnabled)
        {
            cullRenderObjects(camera);

            buildRenderQueue();
            sortRenderQueue();

            if (camera->mColorTarget == ColorTarget::Color || camera->mColorTarget == ColorTarget::ShadowCascades)
            {
                if (camera->mRenderPath == RenderPath::Forward)
                {
                    mForwardRenderer.update(input, camera, mRenderObjects, mModels, mTransformIds, mSpriteObjects);
                }
                else
                {
                    mDeferredRenderer.update(input, camera, mRenderObjects, mModels, mTransformIds);
                }
            }
            else
            {
                mDebugRenderer.update(input, camera, mRenderObjects, mModels, mTransformIds);
            }
        }
    }
}

void RenderSystem::registerRenderAssets(World *world)
{
    // create all texture assets not already created
    for (size_t i = 0; i < world->getNumberOfAssets<Texture2D>(); i++)
    {
        Texture2D *texture = world->getAssetByIndex<Texture2D>(i);
        if (!texture->isCreated())
        {
            texture->create();

            if (!texture->isCreated())
            {
                std::string errorMessage = "Error: Failed to create texture " + texture->getId().toString() + "\n";
                Log::error(errorMessage.c_str());
            }
        }

        if (texture->updateRequired())
        {
            texture->update();
        }
    }

    // create all render texture assets not already created
    for (size_t i = 0; i < world->getNumberOfAssets<RenderTexture>(); i++)
    {
        RenderTexture* texture = world->getAssetByIndex<RenderTexture>(i);
        if (!texture->isCreated())
        {
            texture->create();

            if (!texture->isCreated())
            {
                std::string errorMessage = "Error: Failed to create render texture " + texture->getId().toString() + "\n";
                Log::error(errorMessage.c_str());
            }
        }

        if (texture->updateRequired())
        {
            texture->update();
        }
    }

    // compile all shader assets and configure uniform blocks not already compiled
    std::unordered_set<Guid> shadersCompiledThisFrame;
    for (size_t i = 0; i < world->getNumberOfAssets<Shader>(); i++)
    {
        Shader *shader = world->getAssetByIndex<Shader>(i);

        if (!shader->isCompiled())
        {
            shader->preprocess();
            shader->compile();

            if (!shader->isCompiled())
            {
                std::string errorMessage = "Shader failed to compile " + shader->getName() + " " + shader->getId().toString() + "\n";
                Log::error(&errorMessage[0]);
            }

            shadersCompiledThisFrame.insert(shader->getId());
        }
    }

    // update material on shader change
    for (size_t i = 0; i < world->getNumberOfAssets<Material>(); i++)
    {
        Material *material = world->getAssetByIndex<Material>(i);

        std::unordered_set<Guid>::iterator it = shadersCompiledThisFrame.find(material->getShaderId());

        if (material->hasShaderChanged() || it != shadersCompiledThisFrame.end())
        {
            material->onShaderChanged(); // need to also do this if the shader code changed but the assigned shader
                                              // on the material remained the same!
        }

        if (material->hasTextureChanged())
        {
            material->onTextureChanged();
        }
    }

    // create all mesh assets not already created
    for (size_t i = 0; i < world->getNumberOfAssets<Mesh>(); i++)
    {
        Mesh *mesh = world->getAssetByIndex<Mesh>(i);

        if (!mesh->isCreated())
        {
            mesh->create();

            if (!mesh->isCreated())
            {
                std::string errorMessage = "Error: Failed to create mesh " + mesh->getId().toString() + "\n";
                Log::error(errorMessage.c_str());
            }
        }
    }

    // create all sprite assets not already created
    for (size_t i = 0; i < world->getNumberOfAssets<Sprite>(); i++)
    {
        Sprite* sprite = world->getAssetByIndex<Sprite>(i);

        if (!sprite->isCreated())
        {
            sprite->create();

            if (!sprite->isCreated())
            {
                std::string errorMessage = "Error: Failed to create sprite " + sprite->getId().toString() + "\n";
                Log::error(errorMessage.c_str());
            }
        }
    }
}

void RenderSystem::registerCameras(World *world)
{
    for (size_t i = 0; i < world->getNumberOfComponents<Camera>(); i++)
    {
        Camera *camera = world->getComponentByIndex<Camera>(i);

        if (!camera->isCreated())
        {
            camera->createTargets();
        }
    }
}

void RenderSystem::registerLights(World *world)
{
    for (size_t i = 0; i < world->getNumberOfComponents<Light>(); i++)
    {
        Light *light = world->getComponentByIndex<Light>(i);

        if (!light->isCreated())
        {
            light->createTargets();
        }

        if (light->isShadowMapResolutionChanged())
        {
            light->resizeTargets();
        }
    }
}

void RenderSystem::buildRenderObjectsList(World *world)
{
    mRenderObjects.clear();
    mModels.clear();
    mTransformIds.clear();
    mBoundingSpheres.clear();

    InstanceMap instanceMap;

    // add enabled renderers to render object list
    for (size_t i = 0; i < world->getNumberOfComponents<MeshRenderer>(); i++)
    {
        MeshRenderer *meshRenderer = world->getComponentByIndex<MeshRenderer>(i);

        if (meshRenderer != nullptr && meshRenderer->mEnabled)
        {
            Transform *transform = meshRenderer->getComponent<Transform>();
            Mesh *mesh = world->getAssetById<Mesh>(meshRenderer->getMesh());

            if (transform == nullptr || mesh == nullptr)
            {
                continue;
            }

            glm::mat4 model = transform->getModelMatrix();

            for (int j = 0; j < meshRenderer->mMaterialCount; j++)
            {
                int materialIndex = world->getIndexOf(meshRenderer->getMaterial(j));
                Material *material = world->getAssetByIndex<Material>(materialIndex);

                // could be nullptr if for example we are adding a material to the renderer in the editor
                // but we have not yet actually set the material
                if (material == nullptr)
                {
                    break;
                }

                int shaderIndex = world->getIndexOf(material->getShaderId());

                int subMeshVertexStartIndex = mesh->getSubMeshStartIndex(j);
                int subMeshVertexEndIndex = mesh->getSubMeshEndIndex(j);

                if (material->mEnableInstancing)
                {
                    RenderObject object;
                    object.instanceStart = 0;
                    object.instanceCount = 0;
                    object.materialIndex = materialIndex;
                    object.shaderIndex = shaderIndex;
                    object.start = subMeshVertexStartIndex;
                    object.size = subMeshVertexEndIndex - subMeshVertexStartIndex;
                    object.vao = mesh->getNativeGraphicsVAO();
                    object.vbo = mesh->getNativeGraphicsVBO(MeshVBO::Instance);
                    object.vbo2 = mesh->getNativeGraphicsVBO(MeshVBO::InstanceColor);
                    object.culled = false;
                    object.instanced = true;

                    std::pair<Guid, RenderObject> key = std::make_pair(material->getId(), object);

                    auto it = instanceMap.find(key);
                    if(it != instanceMap.end())
                    {
                        it->second.models.push_back(model);
                        it->second.transformIds.push_back(transform->getId());
                        it->second.boundingSpheres.push_back(Sphere());
                    }
                    else
                    {
                        instanceMap[key].models = std::vector<glm::mat4>();
                        instanceMap[key].transformIds = std::vector<Guid>();
                        instanceMap[key].boundingSpheres = std::vector<Sphere>();
                       
                        instanceMap[key].models.push_back(model);
                        instanceMap[key].transformIds.push_back(transform->getId());
                        //instanceMap[key].boundingSpheres.push_back(Sphere());
                        mBoundingSpheres.push_back(mesh->getBounds());
                    }
                }
                else
                {
                    RenderObject object;
                    object.instanceStart = 0;
                    object.instanceCount = 0;
                    object.materialIndex = materialIndex;
                    object.shaderIndex = shaderIndex;
                    object.start = subMeshVertexStartIndex;
                    object.size = subMeshVertexEndIndex - subMeshVertexStartIndex;
                    object.vao = mesh->getNativeGraphicsVAO();
                    object.vbo = -1;
                    object.vbo2 = -1;
                    object.culled = false;
                    object.instanced = false;

                    mRenderObjects.push_back(object);
                    mModels.push_back(model);
                    mTransformIds.push_back(transform->getId());
                    /*mBoundingSpheres.push_back(Sphere());*/
                    mBoundingSpheres.push_back(mesh->getBounds());
                }
            }
        }
    }

    for (auto it = instanceMap.begin(); it != instanceMap.end(); it++)
    {
        std::vector<glm::mat4> &models = it->second.models;
        std::vector<Guid> &transformIds = it->second.transformIds;
        std::vector<Sphere> &boundingSpheres = it->second.boundingSpheres;

        size_t count = 0;
        while (count < models.size())
        {
            RenderObject renderObject = it->first.second;
            renderObject.instanceStart = count;
            renderObject.instanceCount = std::min(models.size() - count, static_cast<size_t>(Graphics::INSTANCE_BATCH_SIZE));
            
            mRenderObjects.push_back(renderObject);
            for (size_t i = 0; i < renderObject.instanceCount; i++)
            {
                mModels.push_back(models[renderObject.instanceStart + i]);
                mTransformIds.push_back(transformIds[renderObject.instanceStart + i]);
                mBoundingSpheres.push_back(boundingSpheres[renderObject.instanceStart + i]);
            }
       
            count += Graphics::INSTANCE_BATCH_SIZE;
        }
    }

    assert(mModels.size() == mTransformIds.size());
    assert(mModels.size() == mBoundingSpheres.size());

    for (size_t i = 0; i < world->getNumberOfComponents<Terrain>(); i++)
    {
        Terrain *terrain = world->getComponentByIndex<Terrain>(i);

        if (terrain != nullptr)
        {
            Transform *transform = terrain->getComponent<Transform>();

            if (transform == nullptr)
            {
                continue;
            }

            glm::mat4 model = transform->getModelMatrix();

            int materialIndex = world->getIndexOf(terrain->getMaterial());
            Material *material = world->getAssetByIndex<Material>(materialIndex);

            // could be nullptr if for example we are adding a material to the renderer in the editor
            // but we have not yet actually set the material
            if (material == nullptr)
            {
                break;
            }

            int shaderIndex = world->getIndexOf(material->getShaderId());
 
            for (int j = 0; j < terrain->getTotalChunkCount(); j++)
            {
                if (terrain->isChunkEnabled(j))
                {
                    RenderObject object;
                    object.instanceStart = 0;
                    object.instanceCount = 0;
                    object.materialIndex = materialIndex;
                    object.shaderIndex = shaderIndex;
                    object.start = terrain->getChunkStart(j);
                    object.size = terrain->getChunkSize(j);
                    object.vao = terrain->getNativeGraphicsVAO();
                    object.vbo = -1;
                    object.vbo2 = -1;
                    object.culled = false;
                    object.instanced = false;

                    mRenderObjects.push_back(object);
                    mModels.push_back(model);
                    mTransformIds.push_back(transform->getId());
                    mBoundingSpheres.push_back(terrain->getChunkBounds(j));
                }
            }
        }
    }
}

void RenderSystem::buildSpriteObjectsList(World* world)
{
    mSpriteObjects.clear();

    // add enabled renderers to render object list
    for (size_t i = 0; i < world->getNumberOfComponents<SpriteRenderer>(); i++)
    {
        SpriteRenderer* spriteRenderer = world->getComponentByIndex<SpriteRenderer>(i);

        if (spriteRenderer->mEnabled)
        {
            Transform* transform = spriteRenderer->getComponent<Transform>();
            Sprite* sprite = world->getAssetById<Sprite>(spriteRenderer->getSprite());

            if (transform == nullptr || sprite == nullptr)
            {
                continue;
            }

            Texture2D* texture = world->getAssetById<Texture2D>(sprite->getTextureId());

            glm::vec2 size = glm::vec2(100, 100);
            //float rotate = 0.0f;

            glm::mat4 model = transform->getModelMatrix();

            if (spriteRenderer->mFlipX)
            {
                model = glm::rotate(model, glm::radians(180.0f), glm::vec3(0, 1, 0));
            }

            if (spriteRenderer->mFlipY)
            {
                model = glm::rotate(model, glm::radians(180.0f), glm::vec3(1, 0, 0));
            }

            SpriteObject object;
            object.model = model;
            object.color = spriteRenderer->mColor;
            object.vao = sprite->getNativeGraphicsVAO();

            if (texture != nullptr)
            {
                object.texture = texture->getNativeGraphics();
            }
            else
            {
                object.texture = -1;
            }

            mSpriteObjects.push_back(object);
        }
    }
}

void RenderSystem::cullRenderObjects(Camera *camera)
{
    int count = 0;
    int index = 0;
    for (size_t i = 0; i < mRenderObjects.size(); i++)
    {
        if (mRenderObjects[i].instanced)
        {
            for (size_t j = 0; j < mRenderObjects[i].instanceCount; j++)
            {
                glm::vec3 centre = mBoundingSpheres[index].mCentre;
                float radius = mBoundingSpheres[index].mRadius;

                glm::vec4 temp = mModels[index] * glm::vec4(centre.x, centre.y, centre.z, 1.0f);

                Sphere cullingSphere;
                cullingSphere.mCentre = glm::vec3(temp.x, temp.y, temp.z);
                cullingSphere.mRadius = radius;

                if (Intersect::intersect(cullingSphere, camera->getFrustum()))
                {
                    count++;
                }

                index++;
            }
        }
        else
        {
            glm::vec3 centre = mBoundingSpheres[index].mCentre;
            float radius = mBoundingSpheres[index].mRadius;

            glm::vec4 temp = mModels[index] * glm::vec4(centre.x, centre.y, centre.z, 1.0f);

            Sphere cullingSphere;
            cullingSphere.mCentre = glm::vec3(temp.x, temp.y, temp.z);
            cullingSphere.mRadius = radius;

            if (Intersect::intersect(cullingSphere, camera->getFrustum()))
            {
                count++;
            }

            index++;
        }
    }

    std::string message = "Objects in camera frustum count " + std::to_string(count) + "\n";
    Log::info(message.c_str());
}

void RenderSystem::buildRenderQueue()
{
    //mRenderQueue.clear();

    //for (size_t i = 0; i < mRenderObjects.size(); i++)
    //{
    //    // for now dont sort
    //    uint64_t key = i;

    //    mRenderQueue.push_back(std::make_pair(key, (int)i));

    //    //if (!mRenderObjects[i].culled)
    //    //{
    //    //    uint64_t key = 0;

    //    //    uint32_t matIndex = mRenderObjects[i].materialIndex;
    //    //    uint32_t depth = 2342;
    //    //    uint32_t reserved = 112;

    //    //    // [ reserved ][  depth  ][ material index]
    //    //    // [  8-bits  ][ 24-bits ][    32-bits    ]
    //    //    // 64                                     0
    //    //    constexpr uint32_t matMask = 0xFFFFFFFF; // 32 least significant bits mask
    //    //    constexpr uint32_t depthMask = 0xFFFFFF; // 24 least significant bits mask
    //    //    constexpr uint32_t reservedMask = 0xFF;  // 8 least significant bits mask

    //    //    matIndex = matMask & matIndex;
    //    //    depth = depthMask & depth;
    //    //    reserved = reservedMask & reserved;

    //    //    uint64_t temp = 0;
    //    //    key |= ((reserved | temp) << 56);
    //    //    key |= ((depth | temp) << 32); // depth back to front
    //    //    key |= ((matIndex | temp) << 0);

    //    //    mRenderQueue.push_back(std::make_pair(key, (int)i));
    //    //}
    //}
}

void RenderSystem::sortRenderQueue()
{
    // sort render queue from highest priority key to lowest
    //std::sort(mRenderQueue.begin(), mRenderQueue.end(),
    //          [=](std::pair<uint64_t, int> &a, std::pair<uint64_t, int> &b) { return a.first > b.first; });
}