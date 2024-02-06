#include <algorithm>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <random>
#include <chrono>
#include <limits>
#include <unordered_set>
#include <stack>
#include <queue>

#include "../../include/systems/RenderSystem.h"

#include "../../include/core/glm.h"
#include "../../include/core/SerializationYaml.h"
#include "../../include/core/Intersect.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"
#include "../../include/core/Util.h"
#include "../../include/core/Ray.h"

#include "../../include/graphics/DeferredRenderer.h"
#include "../../include/graphics/ForwardRenderer.h"

using namespace PhysicsEngine;

RenderSystem::RenderSystem(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
    mRaytraceEnabled = false;
}

RenderSystem::RenderSystem(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
    mRaytraceEnabled = false;
}

RenderSystem::~RenderSystem()
{
    delete mOcclusionVertexBuffer;
    delete mOcclusionModelIndexBuffer;
    delete mOcclusionIndexBuffer;
    delete mOcclusionMeshHandle;

    delete mOcclusionQuery[0];
    delete mOcclusionQuery[1];

    for (size_t i = 0; i < mBatches.size(); i++)
    {
        delete mBatches[i].mVertexBuffer;
        delete mBatches[i].mNormalBuffer;
        delete mBatches[i].mTexCoordsBuffer;
        delete mBatches[i].mIndexBuffer;
        delete mBatches[i].mMeshHandle;
    }
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
    mRaytracer.init(mWorld);

    mOcclusionVertexBuffer = VertexBuffer::create();
    mOcclusionVertexBuffer->bind(0);
    mOcclusionVertexBuffer->resize(sizeof(float) * 3 * Renderer::MAX_OCCLUDER_VERTEX_COUNT);
    mOcclusionVertexBuffer->unbind(0);

    mOcclusionModelIndexBuffer = VertexBuffer::create();
    mOcclusionModelIndexBuffer->bind(1);
    mOcclusionModelIndexBuffer->resize(sizeof(int) * Renderer::MAX_OCCLUDER_VERTEX_COUNT);
    mOcclusionModelIndexBuffer->unbind(1);

    mOcclusionIndexBuffer = IndexBuffer::create();
    mOcclusionIndexBuffer->bind();
    mOcclusionIndexBuffer->resize(sizeof(unsigned int) * Renderer::MAX_OCCLUDER_INDEX_COUNT);
    mOcclusionIndexBuffer->unbind();

    mOcclusionMeshHandle = MeshHandle::create();
    mOcclusionMeshHandle->addVertexBuffer(mOcclusionVertexBuffer, "POSITION", AttribType::Vec3);
    mOcclusionMeshHandle->addVertexBuffer(mOcclusionModelIndexBuffer, "MODEL_INDEX", AttribType::Int);
    mOcclusionMeshHandle->addIndexBuffer(mOcclusionIndexBuffer);

    mOccluderVertices.resize(3 * Renderer::MAX_OCCLUDER_VERTEX_COUNT);
    mOccluderModelIndices.resize(Renderer::MAX_OCCLUDER_VERTEX_COUNT);
    mOccluderIndices.resize(Renderer::MAX_OCCLUDER_INDEX_COUNT);

    mOcclusionQuery[0] = OcclusionQuery::create();
    mOcclusionQuery[1] = OcclusionQuery::create();
    mOcclusionQueryIndex = -1;

    mBatches.resize(10);
    for (size_t i = 0; i < mBatches.size(); i++)
    {
        mBatches[i].mVertexBuffer = VertexBuffer::create();
        mBatches[i].mNormalBuffer = VertexBuffer::create();
        mBatches[i].mTexCoordsBuffer = VertexBuffer::create();
        mBatches[i].mIndexBuffer = IndexBuffer::create();
        mBatches[i].mMeshHandle = MeshHandle::create();
    }
}

void RenderSystem::update()
{
    registerRenderAssets();

    cacheRenderData();

    buildBVH();
    buildTLAS();

    for (size_t i = 0; i < mWorld->getActiveScene()->getNumberOfComponents<Camera>(); i++)
    {
        Camera *camera = mWorld->getActiveScene()->getComponentByIndex<Camera>(i);

        if (camera->mEnabled)
        {
            Transform *cameraTransform = camera->getComponent<Transform>();
            
            camera->computeViewMatrix(cameraTransform->getPosition(), cameraTransform->getForward(), cameraTransform->getUp(), cameraTransform->getRight());

            frustumCulling(camera);
            occlusionCulling(camera);

            buildRenderQueue();
            sortRenderQueue();

            buildDrawCallCommandList();

            if (camera->mColorTarget == ColorTarget::Color || camera->mColorTarget == ColorTarget::ShadowCascades)
            {
                if (camera->mRenderPath == RenderPath::Forward)
                {
                    mForwardRenderer.update(camera, mDrawCallCommands, mModels, mTransformIds,
                                            *mOcclusionQuery[mOcclusionQueryIndex]);
                }
                else
                {
                    mDeferredRenderer.update(camera, mDrawCallCommands, mModels, mTransformIds);
                }
            }
            else
            {
                mDebugRenderer.update(camera, mDrawCallCommands, mModels, mTransformIds);
            }

            mOcclusionQueryIndex = (mOcclusionQueryIndex == 0) ? 1 : 0;
        
            if (getKeyUp(getInput(), KeyCode::Z))
            {
                mRaytraceEnabled = true;
            }
            else if (getKeyUp(getInput(), KeyCode::X))
            {
                mRaytraceEnabled = false;
            }

            if (mRaytraceEnabled)
            {
                mRaytracer.update(camera, mTLAS, mBLAS, mBVH2, mSpheres);
            }
        }
    }
}

void RenderSystem::registerRenderAssets()
{
    // create all texture assets not already created
    for (size_t i = 0; i < mWorld->getNumberOfAssets<Texture2D>(); i++)
    {
        Texture2D *texture = mWorld->getAssetByIndex<Texture2D>(i);
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
    for (size_t i = 0; i < mWorld->getNumberOfAssets<RenderTexture>(); i++)
    {
        RenderTexture *texture = mWorld->getAssetByIndex<RenderTexture>(i);
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
    for (size_t i = 0; i < mWorld->getNumberOfAssets<Shader>(); i++)
    {
        Shader *shader = mWorld->getAssetByIndex<Shader>(i);

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
    for (size_t i = 0; i < mWorld->getNumberOfAssets<Material>(); i++)
    {
        Material *material = mWorld->getAssetByIndex<Material>(i);

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
    for (size_t i = 0; i < mWorld->getNumberOfAssets<Mesh>(); i++)
    {
        Mesh *mesh = mWorld->getAssetByIndex<Mesh>(i);

        if (mesh->deviceUpdateRequired())
        {
            mesh->copyMeshToDevice();
        }
    }
}

void RenderSystem::cacheRenderData()
{
    size_t meshRendererCount = mWorld->getActiveScene()->getNumberOfComponents<MeshRenderer>();

    mCachedModels.resize(meshRendererCount);
    mCachedTransformIds.resize(meshRendererCount);
    mCachedBoundingSpheres.resize(meshRendererCount);
    mCachedBoundingAABBs.resize(meshRendererCount);
    mCachedMeshIndices.resize(meshRendererCount);
    mCachedMaterialIndices.resize(8 * meshRendererCount);

    for (size_t i = 0; i < meshRendererCount; i++)
    {
        TransformData *transformData = mWorld->getActiveScene()->getTransformDataByMeshRendererIndex(i);
        assert(transformData != nullptr);

        mCachedModels[i] = transformData->getModelMatrix();
    }

    for (size_t i = 0; i < meshRendererCount; i++)
    {
        size_t transformIndex = mWorld->getActiveScene()->getIndexOfTransformFromMeshRendererIndex(i);
        Transform *transform = mWorld->getActiveScene()->getComponentByIndex<Transform>(transformIndex);
        assert(transform != nullptr);

        mCachedTransformIds[i] = transform->getId();
    }

    Id lastMeshId = Id::INVALID;
    int lastMeshIndex = -1;

    Id lastMaterialId = Id::INVALID;
    int lastMaterialIndex = -1;

    glm::vec3 boundingVolumeCentre = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 boundingVolumeMin = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    glm::vec3 boundingVolumeMax = glm::vec3(FLT_MIN, FLT_MIN, FLT_MIN);
    for (size_t i = 0; i < meshRendererCount; i++)
    {
        MeshRenderer *meshRenderer = mWorld->getActiveScene()->getComponentByIndex<MeshRenderer>(i);
        assert(meshRenderer != nullptr);

        Id meshId = meshRenderer->getMeshId();
        if (meshId != lastMeshId)
        {
            lastMeshIndex = mWorld->getIndexOf(meshId);
            lastMeshId = meshId;
        }

        mCachedMeshIndices[i] = lastMeshIndex;

        for (int j = 0; j < meshRenderer->mMaterialCount; j++)
        {
            Id materialId = meshRenderer->getMaterialId(j);
            if (materialId != lastMaterialId)
            {
                lastMaterialIndex = mWorld->getIndexOf(materialId);
                lastMaterialId = materialId;
            }

            mCachedMaterialIndices[8 * i + j] = lastMaterialIndex;
        }

        Mesh *mesh = mWorld->getAssetByIndex<Mesh>(lastMeshIndex);

        mCachedBoundingSpheres[i] = computeWorldSpaceBoundingSphere(mCachedModels[i], mesh->getBounds());

        glm::vec3 centre = mCachedBoundingSpheres[i].mCentre;
        float radius = mCachedBoundingSpheres[i].mRadius;

        boundingVolumeMin.x = glm::min(boundingVolumeMin.x, centre.x - radius);
        boundingVolumeMin.y = glm::min(boundingVolumeMin.y, centre.y - radius);
        boundingVolumeMin.z = glm::min(boundingVolumeMin.z, centre.z - radius);

        boundingVolumeMax.x = glm::max(boundingVolumeMax.x, centre.x + radius);
        boundingVolumeMax.y = glm::max(boundingVolumeMax.y, centre.y + radius);
        boundingVolumeMax.z = glm::max(boundingVolumeMax.z, centre.z + radius);

        mCachedBoundingAABBs[i].mCentre = centre;
        mCachedBoundingAABBs[i].mSize = glm::vec3(2 * radius, 2 * radius, 2 * radius);
    }

    mCachedBoundingVolume.mCentre = 0.5f * (boundingVolumeMax + boundingVolumeMin);
    mCachedBoundingVolume.mSize = (boundingVolumeMax - boundingVolumeMin);
}

void RenderSystem::buildBVH()
{
    if (mCachedBoundingAABBs.size() > 0)
    {
        mBVH.buildBVH(mCachedBoundingAABBs);
    }
}

void RenderSystem::buildTLAS()
{
    /*// Spheres test
    static bool generate_bvh = true;
    if (generate_bvh)
    {
        srand(0);
        int sphereCount = 9;
        mSpheres.resize(sphereCount);
        mSpheres[0] = Sphere(glm::vec3(0.0, -100.5, -1.0f), 100.0f);
        mSpheres[1] = Sphere(glm::vec3(2.0f, 0.0f, -1.0f), 0.5f);
        mSpheres[2] = Sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f);
        mSpheres[3] = Sphere(glm::vec3(-2.0f, 0.0f, -1.0f), 0.5f);
        mSpheres[4] = Sphere(glm::vec3(2.0f, 0.0f, 1.0f), 0.5f);
        mSpheres[5] = Sphere(glm::vec3(0.0f, 0.0f, 1.0f), 0.5f);
        mSpheres[6] = Sphere(glm::vec3(-2.0f, 0.0f, 1.0f), 0.5f);
        mSpheres[7] = Sphere(glm::vec3(0.5f, 1.0f, 0.5f), 0.5f);
        mSpheres[8] = Sphere(glm::vec3(-1.5f, 1.5f, 0.0f), 0.3f);

        std::vector<AABB> boundingVolumes(sphereCount);
        for (int i = 0; i < sphereCount; i++)
        {
            boundingVolumes[i].mCentre = mSpheres[i].mCentre;
            boundingVolumes[i].mSize = 2.0f * glm::vec3(mSpheres[i].mRadius, mSpheres[i].mRadius, mSpheres[i].mRadius);
        }

        mBVH2.buildBVH(boundingVolumes);

        generate_bvh = true;
    }*/








    /*// TLAS and BLAS test
    static bool generate_blas = true;
    if (generate_blas)
    {
        Mesh *planeMesh = mWorld->getPrimtiveMesh(PrimitiveType::Plane);
        Mesh *sphereMesh = mWorld->getPrimtiveMesh(PrimitiveType::Sphere);
        Mesh *cubeMesh = mWorld->getPrimtiveMesh(PrimitiveType::Cube);

        std::vector<float> planeVertices = planeMesh->getVertices();
        std::vector<float> sphereVertices = sphereMesh->getVertices();
        std::vector<float> cubeVertices = cubeMesh->getVertices();

        std::vector<unsigned int> planeIndices = planeMesh->getIndices();
        std::vector<unsigned int> sphereIndices = sphereMesh->getIndices();
        std::vector<unsigned int> cubeIndices = cubeMesh->getIndices();

        std::vector<Triangle> planeTriangles(planeIndices.size() / 3);
        std::vector<Triangle> sphereTriangles(sphereIndices.size() / 3);
        std::vector<Triangle> cubeTriangles(cubeIndices.size() / 3);

        for (size_t i = 0; i < planeIndices.size() / 3; i++)
        {
            unsigned int i0 = planeIndices[3 * i + 0];
            unsigned int i1 = planeIndices[3 * i + 1];
            unsigned int i2 = planeIndices[3 * i + 2];

            glm::vec3 v0 = glm::vec3(planeVertices[3 * i0 + 0], planeVertices[3 * i0 + 1], planeVertices[3 * i0 + 2]);
            glm::vec3 v1 = glm::vec3(planeVertices[3 * i1 + 0], planeVertices[3 * i1 + 1], planeVertices[3 * i1 + 2]);
            glm::vec3 v2 = glm::vec3(planeVertices[3 * i2 + 0], planeVertices[3 * i2 + 1], planeVertices[3 * i2 + 2]);

            planeTriangles[i].mV0 = v0;
            planeTriangles[i].mV1 = v1;
            planeTriangles[i].mV2 = v2;
        }

        for (size_t i = 0; i < sphereIndices.size() / 3; i++)
        {
            unsigned int i0 = sphereIndices[3 * i + 0];
            unsigned int i1 = sphereIndices[3 * i + 1];
            unsigned int i2 = sphereIndices[3 * i + 2];

            glm::vec3 v0 =
                glm::vec3(sphereVertices[3 * i0 + 0], sphereVertices[3 * i0 + 1], sphereVertices[3 * i0 + 2]);
            glm::vec3 v1 =
                glm::vec3(sphereVertices[3 * i1 + 0], sphereVertices[3 * i1 + 1], sphereVertices[3 * i1 + 2]);
            glm::vec3 v2 =
                glm::vec3(sphereVertices[3 * i2 + 0], sphereVertices[3 * i2 + 1], sphereVertices[3 * i2 + 2]);

            sphereTriangles[i].mV0 = v0;
            sphereTriangles[i].mV1 = v1;
            sphereTriangles[i].mV2 = v2;
        }

        for (size_t i = 0; i < cubeIndices.size() / 3; i++)
        {
            unsigned int i0 = cubeIndices[3 * i + 0];
            unsigned int i1 = cubeIndices[3 * i + 1];
            unsigned int i2 = cubeIndices[3 * i + 2];

            glm::vec3 v0 = glm::vec3(cubeVertices[3 * i0 + 0], cubeVertices[3 * i0 + 1], cubeVertices[3 * i0 + 2]);
            glm::vec3 v1 = glm::vec3(cubeVertices[3 * i1 + 0], cubeVertices[3 * i1 + 1], cubeVertices[3 * i1 + 2]);
            glm::vec3 v2 = glm::vec3(cubeVertices[3 * i2 + 0], cubeVertices[3 * i2 + 1], cubeVertices[3 * i2 + 2]);

            cubeTriangles[i].mV0 = v0;
            cubeTriangles[i].mV1 = v1;
            cubeTriangles[i].mV2 = v2;
        }

        glm::mat4 planeModel = glm::mat4(1.0f);
        planeModel[0] *= 10.0f;
        planeModel[1] *= 10.0f;
        planeModel[2] *= 10.0f;
        glm::mat4 sphereModelLeft = glm::mat4(1.0f);
        sphereModelLeft[3] = glm::vec4(-2.0f, 1.0f, 0.0f, 1.0f);
        glm::mat4 sphereModelCentre = glm::mat4(1.0f);
        sphereModelCentre[3] = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
        glm::mat4 sphereModelRight = glm::mat4(1.0f);
        sphereModelRight[3] = glm::vec4(2.0f, 1.0f, 0.0f, 1.0f);
        glm::mat4 cubeModel = glm::mat4(1.0f);
        cubeModel[3] = glm::vec4(0.0f, 3.0f, 0.0f, 1.0f);

        mBLAS.resize(5);
        mBLAS[0].buildBLAS(planeTriangles);
        mBLAS[0].setModel(planeModel);
        mBLAS[1].buildBLAS(sphereTriangles);
        mBLAS[1].setModel(sphereModelLeft);
        mBLAS[2].buildBLAS(sphereTriangles);
        mBLAS[2].setModel(sphereModelCentre);
        mBLAS[3].buildBLAS(sphereTriangles);
        mBLAS[3].setModel(sphereModelRight);
        mBLAS[4].buildBLAS(cubeTriangles);
        mBLAS[4].setModel(cubeModel);

        generate_blas = false;
    }

    mTLAS.buildTLAS(mBLAS);*/





    size_t meshRendererCount = mWorld->getActiveScene()->getNumberOfComponents<MeshRenderer>();

    mBLAS.resize(meshRendererCount);

    for (size_t i = 0; i < meshRendererCount; i++)
    {
        MeshRenderer *meshRenderer = mWorld->getActiveScene()->getComponentByIndex<MeshRenderer>(i);
        Mesh *mesh = mWorld->getAssetById<Mesh>(meshRenderer->getMeshId());

        TransformData *transformData = mWorld->getActiveScene()->getTransformDataByMeshRendererIndex(i);

        mBLAS[i] = mesh->getBLAS();
        mBLAS[i]->setModel(transformData->getModelMatrix());
    }

    mTLAS.buildTLAS(mBLAS);
}

void RenderSystem::frustumCulling(const Camera *camera)
{
    size_t meshRendererCount = mWorld->getActiveScene()->getNumberOfComponents<MeshRenderer>();

    if (meshRendererCount > 0)
    {
        mDistanceToCamera.resize(meshRendererCount);
        mFrustumVisible.resize(meshRendererCount);
        for (size_t i = 0; i < mFrustumVisible.size(); i++)
        {
            mDistanceToCamera[i] = std::pair<float, int>(std::numeric_limits<float>::max(), static_cast<int>(i));
            mFrustumVisible[i] = 0;
        }

        std::queue<int> queue;
        queue.push(0);

        while (!queue.empty())
        {
            int nodeIndex = queue.front();
            queue.pop();

            BVHNode *node = &mBVH.mNodes[nodeIndex];

            AABB aabb;
            aabb.mCentre = 0.5f * (node->mMax + node->mMin);
            aabb.mSize = node->mMax - node->mMin;

            if (Intersect::intersect(aabb, camera->getFrustum()))
            {
                if (node->mIndexCount == 0)
                {
                    queue.push(node->mLeftOrStartIndex);
                    queue.push(node->mLeftOrStartIndex + 1);
                }
                else
                {
                    float distanceToCamera = glm::distance2(aabb.mCentre, camera->getPosition());

                    int startIndex = node->mLeftOrStartIndex;
                    int endIndex = startIndex + node->mIndexCount;
                    for (int i = startIndex; i < endIndex; i++)
                    {
                        mFrustumVisible[mBVH.mPerm[i]] = 1;

                        mDistanceToCamera[mBVH.mPerm[i]] = std::pair<float, int>(distanceToCamera, mBVH.mPerm[i]);
                    }
                }
            }
            else
            {
                mFrustumVisible[mBVH.mPerm[node->mLeftOrStartIndex]] = 0;
            }
        }

        //for (size_t i = 0; i < meshRendererCount; i++)
        //{
        //    mFrustumVisible[i] = Intersect::intersect(mCachedBoundingSpheres[i], camera->getFrustum()); 
        //    //mFrustumVisible[i] = Intersect::intersect(mCachedBoundingAABBs[i], camera->getFrustum());
        //
        //    float distanceToCamera = mFrustumVisible[i]
        //                               ? glm::distance2(mCachedBoundingSpheres[i].mCentre, camera->getPosition())
        //                               : std::numeric_limits<float>::max();

        //    mDistanceToCamera[i] = std::pair<float, int>(distanceToCamera, static_cast<int>(i));
        //}
    }
}

void RenderSystem::occlusionCulling(const Camera *camera)
{
    //size_t meshRendererCount = mWorld->getActiveScene()->getNumberOfComponents<MeshRenderer>();
    //
    //if (meshRendererCount > 0)
    //{
    //    std::sort(mDistanceToCamera.begin(), mDistanceToCamera.end(),
    //              [=](std::pair<float, int> &a, std::pair<float, int> &b) { return a.first < b.first; });

    //    std::array<glm::mat4, 20> models;

    //    size_t cumulativeVertices = 0;
    //    size_t cumulativeIndices = 0;
    //    std::array<int, 20> occluders;

    //    int i = 0;
    //    int cumulativeMeshCount = 0;
    //    while (i < meshRendererCount && cumulativeMeshCount < Renderer::MAX_OCCLUDER_COUNT && cumulativeVertices < Renderer::MAX_OCCLUDER_VERTEX_COUNT && cumulativeIndices < Renderer::MAX_OCCLUDER_INDEX_COUNT)
    //    {
    //        if (mFrustumVisible[i])
    //        {
    //            Mesh *mesh = mWorld->getAssetByIndex<Mesh>(mCachedMeshIndices[i]);
    //            assert(mesh != nullptr);

    //            if ((cumulativeVertices + mesh->getVertexCount()) < Renderer::MAX_OCCLUDER_VERTEX_COUNT)
    //            {
    //                if ((cumulativeIndices + mesh->getIndexCount()) < Renderer::MAX_OCCLUDER_INDEX_COUNT)
    //                {
    //                    const std::vector<float> &vert = mesh->getVertices();
    //                    const std::vector<unsigned int> &ind = mesh->getIndices();

    //                    for (size_t j = 0; j < mesh->getVertexCount(); j++)
    //                    {
    //                        mOccluderVertices[3 * cumulativeVertices + 3 * j + 0] = vert[3 * j + 0];
    //                        mOccluderVertices[3 * cumulativeVertices + 3 * j + 1] = vert[3 * j + 1];
    //                        mOccluderVertices[3 * cumulativeVertices + 3 * j + 2] = vert[3 * j + 2];
    //                    
    //                        mOccluderModelIndices[cumulativeVertices + j] = cumulativeMeshCount;
    //                    }

    //                    for (size_t j = 0; j < mesh->getIndexCount(); j++)
    //                    {
    //                        mOccluderIndices[cumulativeIndices + j] = cumulativeVertices + ind[j];
    //                    }

    //                    models[cumulativeMeshCount] = mCachedModels[i];

    //                    occluders[cumulativeMeshCount] = i;
    //                    cumulativeVertices += mesh->getVertexCount();
    //                    cumulativeIndices += mesh->getIndexCount();
    //                    cumulativeMeshCount++;
    //                }
    //            }
    //        }
    //        i++;
    //    }

    //    mOcclusionVertexBuffer->bind();
    //    mOcclusionVertexBuffer->setData(mOccluderVertices.data(), 0, sizeof(float) * 3 * cumulativeVertices);
    //    mOcclusionVertexBuffer->unbind();

    //    mOcclusionModelIndexBuffer->bind();
    //    mOcclusionModelIndexBuffer->setData(mOccluderModelIndices.data(), 0, sizeof(int) * cumulativeVertices);
    //    mOcclusionModelIndexBuffer->unbind();

    //    mOcclusionIndexBuffer->bind();
    //    mOcclusionIndexBuffer->setData(mOccluderIndices.data(), 0, sizeof(unsigned int) * cumulativeIndices);
    //    mOcclusionIndexBuffer->unbind();

    //    mOcclusionMapShader = RendererShaders::getOcclusionMapShader();

    //    mOcclusionModelMatUniform = RendererUniforms::getOcclusionUniform();
    //    for (int j = 0; j < models.size(); j++)
    //    {
    //        mOcclusionModelMatUniform->setModel(models[j], j);   
    //    }
    //    mOcclusionModelMatUniform->copyToUniformsToDevice();

    //    // Drawing
    //    camera->getNativeGraphicsOcclusionMapFBO()->bind();
    //    Renderer::getRenderer()->setViewport(0, 0, camera->getNativeGraphicsOcclusionMapFBO()->getWidth(), camera->getNativeGraphicsOcclusionMapFBO()->getHeight());
    //    camera->getNativeGraphicsOcclusionMapFBO()->clearColor(Color::pink);

    //    Renderer::getRenderer()->turnOff(Capability::Depth_Testing);

    //    mOcclusionMapShader->bind();
    //    mOcclusionMapShader->setView(camera->getViewMatrix());
    //    mOcclusionMapShader->setProjection(camera->getProjMatrix());
    //    mOcclusionMeshHandle->drawIndexed(0, cumulativeIndices);
    //    mOcclusionMapShader->unbind();

    //    Renderer::getRenderer()->turnOn(Capability::Depth_Testing);

    //    camera->getNativeGraphicsOcclusionMapFBO()->unbind();
    //}

    /*size_t meshRendererCount = mWorld->getActiveScene()->getNumberOfComponents<MeshRenderer>();

    if (meshRendererCount > 0)
    {
        mOcclusionQuery[0]->increaseQueryCount(meshRendererCount);
        mOcclusionQuery[1]->increaseQueryCount(meshRendererCount);

        if (mOcclusionQueryIndex != -1)
        {
            int lastFramesOcclusionQueryIndex = (mOcclusionQueryIndex == 0) ? 1 : 0;

            for (size_t i = 0; i < mFrustumVisible.size(); i++)
            {
                if (mFrustumVisible[i])
                {
                    if (!mOcclusionQuery[lastFramesOcclusionQueryIndex]->isVisible(i))
                    {
                        std::cout << "i: " << std::to_string(i) << " not visble" << std::endl;
                    }
                }
            }
        }
        else
        {
            mOcclusionQueryIndex = 0;
        }
    }*/
}

void RenderSystem::buildRenderQueue()
{
    size_t meshRendererCount = mWorld->getActiveScene()->getNumberOfComponents<MeshRenderer>();

    // allow up to 8 materials (and hence draw calls) per mesh renderer
    mRenderQueueScratch.resize(8 * meshRendererCount);

    size_t drawCallCount = 0;

    for (size_t i = 0; i < meshRendererCount; i++)
    {
        if (mFrustumVisible[i])
        {
            MeshRenderer *meshRenderer = mWorld->getActiveScene()->getComponentByIndex<MeshRenderer>(i);
            assert(meshRenderer != nullptr);

            if (meshRenderer->mEnabled)
            {
                // could be nullptr if for example we are adding a mesh to the renderer in the editor
                // but we have not yet actually set the mesh
                if (mCachedMeshIndices[i] != -1)
                {
                    for (int j = 0; j < meshRenderer->mMaterialCount; j++)
                    {
                        int materialIndex = mCachedMaterialIndices[8 * i + j];
                        Material *material = mWorld->getAssetByIndex<Material>(materialIndex);

                        // could be nullptr if for example we are adding a material to the renderer in the editor
                        // but we have not yet actually set the material
                        if (material != nullptr)
                        {
                            DrawCallCommand command;
                            command.generateDrawCall(materialIndex, mCachedMeshIndices[i], j, 0, 0);
                            
                            mRenderQueueScratch[drawCallCount].first = command;
                            mRenderQueueScratch[drawCallCount].second = static_cast<int>(i);

                            drawCallCount++;
                        }
                        else
                        {
                            break;
                        }
                    }
                }
            }
        }
    }

    mRenderQueue.resize(drawCallCount);

    for (size_t i = 0; i < drawCallCount; i++)
    {
        mRenderQueue[i] = mRenderQueueScratch[i];
    }

    assert(mCachedBoundingSpheres.size() == mFrustumVisible.size());
    assert(mCachedBoundingAABBs.size() == mFrustumVisible.size());

    mWorld->mBoundingSpheres = mCachedBoundingSpheres;
    mWorld->mBoundingAABBs = mCachedBoundingAABBs;
    mWorld->mFrustumVisible = mFrustumVisible;
    mWorld->mBoundingVolume = mCachedBoundingVolume;
}

void RenderSystem::sortRenderQueue()
{
    // sort render queue from highest priority key to lowest
    std::sort(mRenderQueue.begin(), mRenderQueue.end(),
              [=](std::pair<DrawCallCommand, int> &a, std::pair<DrawCallCommand, int> &b) {
                  return a.first.getCode() > b.first.getCode();
              });

    mModels.resize(mRenderQueue.size());
    mTransformIds.resize(mRenderQueue.size());
    mBoundingSpheres.resize(mRenderQueue.size());

    for (size_t i = 0; i < mRenderQueue.size(); i++)
    {
        int j = mRenderQueue[i].second;

        mModels[i] = mCachedModels[j];
        mTransformIds[i] = mCachedTransformIds[j];
        mBoundingSpheres[i] = mCachedBoundingSpheres[j];
    }
}

void RenderSystem::buildDrawCallCommandList()
{
    mDrawCallCommands.resize(mRenderQueue.size());

    // Iterate through sorted render queue and build draw call commands 
    // that can be: single draw calls, instanced draw calls, or batched draw calls.
    // These draw call commands will then be passed to the actual renderer.
    int drawCallIndex = 0;

    int index = 0;
    while (index < mRenderQueue.size())
    {
        DrawCallCommand command(mRenderQueue[index].first);

        uint16_t materialIndex = command.getMaterialIndex();
        uint16_t meshIndex = command.getMeshIndex();
        uint8_t subMesh = command.getSubMesh();

        // See if we can use instancing
        bool instanced = false;
        if (index + Renderer::INSTANCE_BATCH_SIZE < mRenderQueue.size())
        {
            DrawCallCommand commandAhead(mRenderQueue[index + Renderer::INSTANCE_BATCH_SIZE].first);
            if (materialIndex == commandAhead.getMaterialIndex())
            {
                if (meshIndex == commandAhead.getMeshIndex())
                {
                    if (subMesh == commandAhead.getSubMesh())
                    {
                        instanced = true;
                    }
                }
            }
        }

        // If not using instancing, check if batching
        int batchCount = 0;
        int batchVertexCount = 0;
        if (!instanced)
        {
            while ((index + batchCount) < mRenderQueue.size())
            {
                if (batchCount >= Renderer::MAX_MESH_PER_BATCH)
                {
                    break;
                }

                if (batchVertexCount >= Renderer::MAX_VERTICES_PER_BATCH)
                {
                    break;
                }

                DrawCallCommand commandAhead(mRenderQueue[index + batchCount].first);
                if (materialIndex == commandAhead.getMaterialIndex())
                {
                    Mesh *mesh = mWorld->getAssetByIndex<Mesh>(commandAhead.getMeshIndex());
                    if (batchVertexCount + mesh->getVertexCount() < Renderer::MAX_VERTICES_PER_BATCH)
                    {
                        batchCount++;
                        batchVertexCount += mesh->getVertexCount();
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    break;
                }
            }
        }

        //std::cout << "batchCount: " << batchCount << " batchVertexCount: " << batchVertexCount << std::endl;

        if (instanced)
        {
            // Can use instanced draw calls
            command.markDrawCallAsInstanced();
            command.markDrawCallAsIndexed();

            mDrawCallCommands[drawCallIndex] = command;

            index += Renderer::INSTANCE_BATCH_SIZE;
        }
        //else if (batchCount > 0)
        //{
        //    
        //    index += batchCount;
        //}
        else
        {
            command.markDrawCallAsIndexed();

            mDrawCallCommands[drawCallIndex] = command;
        
            index++;
        }

        drawCallIndex++;
    }

    assert(index == mRenderQueue.size());

    mDrawCallCommands.resize(drawCallIndex);

    // Note: Not really efficient but ok for now. Adds enabled terrain to draw call command list
    for (size_t i = 0; i < mWorld->getActiveScene()->getNumberOfComponents<Terrain>(); i++)
    {
         Terrain *terrain = mWorld->getActiveScene()->getComponentByIndex<Terrain>(i);
    
         Transform *transform = terrain->getComponent<Transform>();
    
         glm::mat4 model = transform->getModelMatrix();
    
         int materialIndex = mWorld->getIndexOf(terrain->getMaterial());
         Material *material = mWorld->getAssetByIndex<Material>(materialIndex);
    
         // could be nullptr if for example we are adding a material to the renderer in the editor
         // but we have not yet actually set the material
         if (material != nullptr)
         {
             for (int j = 0; j < terrain->getTotalChunkCount(); j++)
             {
                 if (terrain->isChunkEnabled(j))
                 {
                     DrawCallCommand command;
                     command.generateTerrainDrawCall(materialIndex, i, j, 0);
                     command.markDrawCallAsTerrain();

                     //DrawCallCommand command;
                     //command.meshHandle = terrain->getNativeGraphicsHandle();
                     //command.instanceModelBuffer = nullptr;
                     //command.instanceColorBuffer = nullptr;
                     //command.material = material;
                     //command.shader = shader;
                     //command.meshStartIndex = (int)terrain->getChunkStart(j);
                     //command.meshEndIndex = (int)(terrain->getChunkStart(j) + terrain->getChunkSize(j));
                     //command.instanceCount = 0;
                     //command.indexed = false;

                    // Not really efficient but ok for now
                    mDrawCallCommands.push_back(command);
                    mModels.push_back(model);
                    mTransformIds.push_back(transform->getId());
                    mBoundingSpheres.push_back(computeWorldSpaceBoundingSphere(model, terrain->getChunkBounds(j)));
                }
            }
        }
    }
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

const BVH &RenderSystem::getBVH() const
{
    return mBVH;
}