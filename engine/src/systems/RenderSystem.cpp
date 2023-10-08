#include <algorithm>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <random>
#include <chrono>
#include <limits>
#include <unordered_set>

#define GLM_FORCE_RADIANS

#include "glm/gtc/matrix_transform.hpp"

#include "../../include/systems/RenderSystem.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/Intersect.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"

#include "../../include/graphics/DeferredRenderer.h"
#include "../../include/graphics/ForwardRenderer.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

RenderSystem::RenderSystem(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

RenderSystem::~RenderSystem()
{
}

void RenderSystem::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;
}

void RenderSystem::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");
}

int RenderSystem::getType() const
{
    return PhysicsEngine::RENDERSYSTEM_TYPE;
}

std::string RenderSystem::getObjectName() const
{
    return PhysicsEngine::RENDERSYSTEM_NAME;
}

Guid RenderSystem::getGuid() const
{
    return mGuid;
}

Id RenderSystem::getId() const
{
    return mId;
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

    buildRenderObjectsList(mWorld);

    for (size_t i = 0; i < mWorld->getActiveScene()->getNumberOfComponents<Camera>(); i++)
    {
        Camera *camera = mWorld->getActiveScene()->getComponentByIndex<Camera>(i);

        if (camera->mEnabled)
        {
            cullRenderObjects(camera);

            buildRenderQueue();
            sortRenderQueue();

            if (camera->mColorTarget == ColorTarget::Color || camera->mColorTarget == ColorTarget::ShadowCascades)
            {
                if (camera->mRenderPath == RenderPath::Forward)
                {
                    mForwardRenderer.update(input, camera, mRenderObjects, mModels, mTransformIds);
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
        if (texture->deviceUpdateRequired())
        {
            texture->copyTextureToDevice();
        }

        if (texture->updateRequired())
        {
            texture->updateTextureParameters();
        }
    }

    // create all render texture assets not already created
    for (size_t i = 0; i < world->getNumberOfAssets<RenderTexture>(); i++)
    {
        RenderTexture *texture = world->getAssetByIndex<RenderTexture>(i);
        if (texture->deviceUpdateRequired())
        {
            texture->copyTextureToDevice();
        }

        if (texture->updateRequired())
        {
            texture->updateTextureParameters();
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
                std::string errorMessage =
                    "Shader failed to compile " + shader->mName + " " + shader->getGuid().toString() + "\n";
                Log::error(&errorMessage[0]);
            }

            shadersCompiledThisFrame.insert(shader->getGuid());
        }
    }

    // update material on shader change
    for (size_t i = 0; i < world->getNumberOfAssets<Material>(); i++)
    {
        Material *material = world->getAssetByIndex<Material>(i);

        std::unordered_set<Guid>::iterator it = shadersCompiledThisFrame.find(material->getShaderGuid());

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

        if (mesh->deviceUpdateRequired())
        {
            mesh->copyMeshToDevice();
        }
    }
}

void RenderSystem::buildRenderObjectsList(World *world)
{
    size_t meshRendererCount = world->getActiveScene()->getNumberOfComponents<MeshRenderer>();

    std::vector<glm::mat4> models;
    models.resize(meshRendererCount);

    std::vector<glm::vec3> scales;
    scales.resize(meshRendererCount);

    //auto startTime = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < meshRendererCount; i++)
    {
        TransformData *transformData = world->getActiveScene()->getTransformDataByMeshRendererIndex(i);
        assert(transformData != nullptr);
    
        models[i] = transformData->getModelMatrix();
        scales[i] = transformData->mScale;
    }

    //auto stopTime = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stopTime - startTime);
    //double gbytes = ((sizeof(TransformData) + sizeof(glm::mat4)) * meshRendererCount) / (1024.0 * 1024.0 * 1024.0);
    //std::cout << gbytes / (duration.count() / 1000000.0) << "\n";

    std::vector<Id> transformIds;
    transformIds.resize(meshRendererCount);

    for (size_t i = 0; i < meshRendererCount; i++)
    {
        size_t transformIndex = world->getActiveScene()->getIndexOfTransformFromMeshRendererIndex(i);
        Transform *transform = world->getActiveScene()->getComponentByIndex<Transform>(transformIndex);
        assert(transform != nullptr);

        transformIds[i] = transform->getId();
    }

    std::vector<Sphere> boundingSpheres;
    boundingSpheres.resize(meshRendererCount);
    
    std::vector<int> meshIndices;
    meshIndices.resize(meshRendererCount);
    
    Id lastMeshId = Id::INVALID;
    int lastMeshIndex = -1;
    for (size_t i = 0; i < meshRendererCount; i++)
    {
        MeshRenderer *meshRenderer = world->getActiveScene()->getComponentByIndex<MeshRenderer>(i);
        assert(meshRenderer != nullptr);

        if (meshRenderer->mEnabled)
        {
            if (meshRenderer->getMeshId() != lastMeshId)
            {
                lastMeshIndex = world->getIndexOf(meshRenderer->getMeshId());
                lastMeshId = meshRenderer->getMeshId();
            }

            Mesh *mesh = world->getAssetByIndex<Mesh>(lastMeshIndex);

            meshIndices[i] = lastMeshIndex;
            boundingSpheres[i] = computeWorldSpaceBoundingSphere(models[i], scales[i], mesh->getBounds());
        }
    }

    std::unordered_map<uint64_t, std::vector<size_t>> instanceMapping;

    std::vector<uint64_t> singleDrawCallKeys(meshRendererCount);
    std::vector<size_t> singleDrawCallMeshRendererIndices(meshRendererCount);

    size_t singleDrawCallCount = 0;

    for (size_t i = 0; i < meshRendererCount; i++)
    {
        MeshRenderer *meshRenderer = world->getActiveScene()->getComponentByIndex<MeshRenderer>(i);
        assert(meshRenderer != nullptr);

        if (meshRenderer->mEnabled)
        {
            // could be nullptr if for example we are adding a mesh to the renderer in the editor
            // but we have not yet actually set the mesh
            if (meshIndices[i] != -1)
            {
                for (int j = 0; j < meshRenderer->mMaterialCount; j++)
                {
                    int materialIndex = world->getIndexOf(meshRenderer->getMaterialId(j));
                    Material *material = world->getAssetByIndex<Material>(materialIndex);

                    // could be nullptr if for example we are adding a material to the renderer in the editor
                    // but we have not yet actually set the material
                    if (material != nullptr)
                    {
                        int shaderIndex = world->getIndexOf(material->getShaderGuid());

                        if (material->mEnableInstancing)
                        {
                            uint64_t key = generateKey(materialIndex, meshIndices[i], shaderIndex, j, 2);

                            auto it = instanceMapping.find(key);
                            if (it != instanceMapping.end())
                            {
                                it->second.push_back(i);
                            }
                            else
                            {
                                instanceMapping[key] = std::vector<size_t>();
                                instanceMapping[key].reserve(Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
                                instanceMapping[key].push_back(i);
                            }
                        }
                        else
                        {
                            singleDrawCallKeys[singleDrawCallCount] =
                                generateKey(materialIndex, meshIndices[i], shaderIndex, j, 1);
                            singleDrawCallMeshRendererIndices[singleDrawCallCount] = i;
                            singleDrawCallCount++;
                        }
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
    }

    singleDrawCallKeys.resize(singleDrawCallCount);

    mTotalRenderObjects.resize(singleDrawCallCount);
    mTotalModels.resize(singleDrawCallCount);
    mTotalTransformIds.resize(singleDrawCallCount);
    mTotalBoundingSpheres.resize(singleDrawCallCount);

    // non-instance draw calls
    for (size_t i = 0; i < singleDrawCallCount; i++)
    {
        RenderObject object;
        object.instanceStart = 0;
        object.instanceCount = 0;
        object.key = singleDrawCallKeys[i];

        mTotalRenderObjects[i] = object;
        mTotalModels[i] = models[singleDrawCallMeshRendererIndices[i]];
        mTotalTransformIds[i] = transformIds[singleDrawCallMeshRendererIndices[i]];
        mTotalBoundingSpheres[i] = boundingSpheres[singleDrawCallMeshRendererIndices[i]];
    }

    // instanced draw calls
    for (auto it = instanceMapping.begin(); it != instanceMapping.end(); it++)
    {
        RenderObject object;
        object.key = it->first;

        size_t count = 0;
        while (count < it->second.size())
        {
            object.instanceStart = count;
            object.instanceCount =
                std::min(it->second.size() - count, static_cast<size_t>(Renderer::getRenderer()->INSTANCE_BATCH_SIZE));

            mTotalRenderObjects.push_back(object);

            size_t start = mTotalModels.size();
            mTotalModels.resize(mTotalModels.size() + object.instanceCount);
            mTotalTransformIds.resize(mTotalTransformIds.size() + object.instanceCount);
            mTotalBoundingSpheres.resize(mTotalBoundingSpheres.size() + object.instanceCount);

            for (size_t i = 0; i < object.instanceCount; i++)
            {
                mTotalModels[start + i] = models[it->second[object.instanceStart + i]];
                mTotalTransformIds[start + i] = transformIds[it->second[object.instanceStart + i]];
                mTotalBoundingSpheres[start + i] = boundingSpheres[it->second[object.instanceStart + i]];
            }

            count += Renderer::getRenderer()->INSTANCE_BATCH_SIZE;
        }
    }




    



















































    //mTotalRenderObjects.clear();
    //mTotalModels.clear();
    //mTotalTransformIds.clear();
    //mTotalBoundingSpheres.clear();

    //InstanceMap instanceMap;

    //// add enabled renderers to render object list
    //for (size_t i = 0; i < world->getActiveScene()->getNumberOfComponents<MeshRenderer>(); i++)
    //{
    //    MeshRenderer *meshRenderer = world->getActiveScene()->getComponentByIndex<MeshRenderer>(i);

    //    assert(meshRenderer != nullptr);

    //    if (meshRenderer->mEnabled)
    //    {
    //        Transform *transform = meshRenderer->getComponent<Transform>();
    //        Mesh *mesh = world->getAssetById<Mesh>(meshRenderer->getMeshId());

    //        assert(transform != nullptr);

    //        // could be nullptr if for example we are adding a mesh to the renderer in the editor
    //        // but we have not yet actually set the mesh
    //        if (mesh == nullptr)
    //        {
    //            continue;
    //        }

    //        glm::mat4 model = transform->getModelMatrix();

    //        Sphere boundingSphere = computeWorldSpaceBoundingSphere(model, mesh->getBounds());

    //        for (int j = 0; j < meshRenderer->mMaterialCount; j++)
    //        {
    //            int materialIndex = world->getIndexOf(meshRenderer->getMaterialId(j));
    //            Material *material = world->getAssetByIndex<Material>(materialIndex);

    //            // could be nullptr if for example we are adding a material to the renderer in the editor
    //            // but we have not yet actually set the material
    //            if (material == nullptr)
    //            {
    //                break;
    //            }

    //            int shaderIndex = world->getIndexOf(material->getShaderGuid());

    //            int subMeshVertexStartIndex = mesh->getSubMeshStartIndex(j);
    //            int subMeshVertexEndIndex = mesh->getSubMeshEndIndex(j);

    //            if (material->mEnableInstancing)
    //            {
    //                RenderObject object;
    //                object.instanceStart = 0;
    //                object.instanceCount = 0;
    //                object.materialIndex = materialIndex;
    //                object.shaderIndex = shaderIndex;
    //                object.start = subMeshVertexStartIndex;
    //                object.size = subMeshVertexEndIndex - subMeshVertexStartIndex;
    //                object.meshHandle = mesh->getNativeGraphicsHandle();
    //                object.instanceModelBuffer = mesh->getNativeGraphicsInstanceModelBuffer();
    //                object.instanceColorBuffer = mesh->getNativeGraphicsInstanceColorBuffer();
    //                object.instanced = true;
    //                object.indexed = false;

    //                std::pair<Guid, RenderObject> key = std::make_pair(material->getGuid(), object);

    //                auto it = instanceMap.find(key);
    //                if (it != instanceMap.end())
    //                {
    //                    it->second.models.push_back(model);
    //                    it->second.transformIds.push_back(transform->getId());
    //                    it->second.boundingSpheres.push_back(boundingSphere);
    //                }
    //                else
    //                {
    //                    instanceMap[key].models = std::vector<glm::mat4>();
    //                    instanceMap[key].transformIds = std::vector<Id>();
    //                    instanceMap[key].boundingSpheres = std::vector<Sphere>();

    //                    instanceMap[key].models.reserve(Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
    //                    instanceMap[key].transformIds.reserve(Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
    //                    instanceMap[key].boundingSpheres.reserve(Renderer::getRenderer()->INSTANCE_BATCH_SIZE);

    //                    instanceMap[key].models.push_back(model);
    //                    instanceMap[key].transformIds.push_back(transform->getId());
    //                    instanceMap[key].boundingSpheres.push_back(boundingSphere);
    //                }
    //            }
    //            else
    //            {
    //                RenderObject object;
    //                object.instanceStart = 0;
    //                object.instanceCount = 0;
    //                object.materialIndex = materialIndex;
    //                object.shaderIndex = shaderIndex;
    //                object.start = subMeshVertexStartIndex;
    //                object.size = subMeshVertexEndIndex - subMeshVertexStartIndex;
    //                object.meshHandle = mesh->getNativeGraphicsHandle();
    //                object.instanceModelBuffer = nullptr;
    //                object.instanceColorBuffer = nullptr;
    //                object.instanced = false;
    //                object.indexed = true;

    //                mTotalRenderObjects.push_back(object);
    //                mTotalModels.push_back(model);
    //                mTotalTransformIds.push_back(transform->getId());
    //                mTotalBoundingSpheres.push_back(boundingSphere);
    //            }
    //        }
    //    }
    //}

    //for (auto it = instanceMap.begin(); it != instanceMap.end(); it++)
    //{
    //    std::vector<glm::mat4> &models = it->second.models;
    //    std::vector<Id> &transformIds = it->second.transformIds;
    //    std::vector<Sphere> &boundingSpheres = it->second.boundingSpheres;

    //    size_t count = 0;
    //    while (count < models.size())
    //    {
    //        RenderObject renderObject = it->first.second;
    //        renderObject.instanceStart = count;
    //        renderObject.instanceCount =
    //            std::min(models.size() - count, static_cast<size_t>(Renderer::getRenderer()->INSTANCE_BATCH_SIZE));

    //        mTotalRenderObjects.push_back(renderObject);

    //        size_t start = mTotalModels.size();
    //        mTotalModels.resize(mTotalModels.size() + renderObject.instanceCount);
    //        mTotalTransformIds.resize(mTotalTransformIds.size() + renderObject.instanceCount);
    //        mTotalBoundingSpheres.resize(mTotalBoundingSpheres.size() + renderObject.instanceCount);

    //        for (size_t i = 0; i < renderObject.instanceCount; i++)
    //        {
    //            //mTotalModels.push_back(models[renderObject.instanceStart + i]);
    //            //mTotalTransformIds.push_back(transformIds[renderObject.instanceStart + i]);
    //            //mTotalBoundingSpheres.push_back(boundingSpheres[renderObject.instanceStart + i]);
    //            mTotalModels[start + i] = models[renderObject.instanceStart + i];
    //            mTotalTransformIds[start + i] = transformIds[renderObject.instanceStart + i];
    //            mTotalBoundingSpheres[start + i] = boundingSpheres[renderObject.instanceStart + i];
    //        }

    //        count += Renderer::getRenderer()->INSTANCE_BATCH_SIZE;
    //    }
    //}

    assert(mTotalModels.size() == mTotalTransformIds.size());
    assert(mTotalModels.size() == mTotalBoundingSpheres.size());

    // add enabled terrain to render object list
    /*for (size_t i = 0; i < world->getActiveScene()->getNumberOfComponents<Terrain>(); i++)
    {
        Terrain *terrain = world->getActiveScene()->getComponentByIndex<Terrain>(i);

        Transform *transform = terrain->getComponent<Transform>();

        glm::mat4 model = transform->getModelMatrix();

        int materialIndex = world->getIndexOf(terrain->getMaterial());
        Material *material = world->getAssetByIndex<Material>(materialIndex);

        // could be nullptr if for example we are adding a material to the renderer in the editor
        // but we have not yet actually set the material
        if (material == nullptr)
        {
            break;
        }

        int shaderIndex = world->getIndexOf(material->getShaderGuid());

        for (int j = 0; j < terrain->getTotalChunkCount(); j++)
        {
            if (terrain->isChunkEnabled(j))
            {
                RenderObject object;
                object.key = generateKey(materialIndex, shaderIndex, );
                object.instanceStart = 0;
                object.instanceCount = 0;
                //object.materialIndex = materialIndex;
                //object.shaderIndex = shaderIndex;
                //object.start = terrain->getChunkStart(j);
                //object.size = terrain->getChunkSize(j);
                //object.meshHandle = terrain->getNativeGraphicsHandle();
                //object.instanceModelBuffer = nullptr;
                //object.instanceColorBuffer = nullptr;
                //object.instanced = false;
                //object.indexed = false;

                mTotalRenderObjects.push_back(object);
                mTotalModels.push_back(model);
                mTotalTransformIds.push_back(transform->getId());

                mTotalBoundingSpheres.push_back(computeWorldSpaceBoundingSphere(model, terrain->getChunkBounds(j)));
            }
        }
    }*/

    assert(mTotalModels.size() == mTotalTransformIds.size());
    assert(mTotalModels.size() == mTotalBoundingSpheres.size());

    mWorld->mBoundingSpheres = mTotalBoundingSpheres;
}

void RenderSystem::cullRenderObjects(Camera *camera)
{
    mModels.resize(mTotalModels.size());
    mTransformIds.resize(mTotalTransformIds.size());
    mRenderObjects.resize(mTotalRenderObjects.size());

    int index = 0;
    int objectCount = 0;
    int count = 0;
    for (size_t i = 0; i < mTotalRenderObjects.size(); i++)
    {
        // dont perform any culling on instanced objects
        if (isInstanced(mTotalRenderObjects[i].key))
        {
            mRenderObjects[objectCount] = mTotalRenderObjects[i];
            mRenderObjects[objectCount].instanceStart = count;

            for (size_t j = 0; j < mTotalRenderObjects[i].instanceCount; j++)
            {
                mModels[count + j] = mTotalModels[index];
                mTransformIds[count + j] = mTotalTransformIds[index];

                index++;
            }

            objectCount++;
            count += mTotalRenderObjects[i].instanceCount;
        }
        else
        {
            if (Intersect::intersect(mTotalBoundingSpheres[index], camera->getFrustum()))
            {
                mRenderObjects[objectCount] = mTotalRenderObjects[i];

                mModels[count] = mTotalModels[index];
                mTransformIds[count] = mTotalTransformIds[index];

                objectCount++;
                count++;
            }

            index++;
        }
    }

    mModels.resize(count);
    mTransformIds.resize(count);
    mRenderObjects.resize(objectCount);
}

void RenderSystem::buildRenderQueue()
{
    // mRenderQueue.clear();

    // for (size_t i = 0; i < mTotalRenderObjects.size(); i++)
    //{
    //     // for now dont sort
    //     uint64_t key = i;

    //    mRenderQueue.push_back(std::make_pair(key, (int)i));

    //    //if (!mTotalRenderObjects[i].culled)
    //    //{
    //    //    uint64_t key = 0;

    //    //    uint32_t matIndex = mTotalRenderObjects[i].materialIndex;
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
    // std::sort(mRenderQueue.begin(), mRenderQueue.end(),
    //          [=](std::pair<uint64_t, int> &a, std::pair<uint64_t, int> &b) { return a.first > b.first; });
}


// should work only if matrix is calculated as M = T * R * S
glm::vec3 extractScale2(const glm::mat4 &m)
{
    glm::vec3 scale;

    scale.x = glm::length2(glm::vec3(m[0][0], m[0][1], m[0][2]));
    scale.y = glm::length2(glm::vec3(m[1][0], m[1][1], m[1][2]));
    scale.z = glm::length2(glm::vec3(m[2][0], m[2][1], m[2][2]));

    return scale;
}

float extractLargestScale(const glm::mat4 &m)
{
    glm::vec3 scale2 = extractScale2(m);

    return glm::sqrt(glm::max(scale2.x, glm::max(scale2.y, scale2.z)));
}

Sphere RenderSystem::computeWorldSpaceBoundingSphere(const glm::mat4 &model, const Sphere &sphere)
{
    Sphere boundingSphere;
    boundingSphere.mCentre = glm::vec3(model * glm::vec4(sphere.mCentre, 1.0f));
    boundingSphere.mRadius = sphere.mRadius * extractLargestScale(model);

    return boundingSphere;
}

Sphere RenderSystem::computeWorldSpaceBoundingSphere(const glm::mat4 &model, const glm::vec3 &scale, const Sphere &sphere)
{
    // M = T * R * S
    // Rxx*Sx Ryx*Sy Rzx*Sz Tx
    // Rxy*Sx Ryy*Sy Rzy*Sz Ty
    // Rxz*Sx Ryz*Sy Rzz*Sz Tz
    //      0      0      0 1

    Sphere boundingSphere;
    boundingSphere.mCentre.x = scale.x * sphere.mCentre.x + model[3][0];
    boundingSphere.mCentre.y = scale.y * sphere.mCentre.y + model[3][1];
    boundingSphere.mCentre.z = scale.z * sphere.mCentre.z + model[3][2];
    //boundingSphere.mCentre = glm::vec3(model * glm::vec4(sphere.mCentre, 1.0f));
    boundingSphere.mRadius = sphere.mRadius * glm::max(scale.x, glm::max(scale.y, scale.z));

    return boundingSphere;
}