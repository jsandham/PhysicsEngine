#include "../../include/components/Terrain.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/World.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Renderer.h"

#include "stb_perlin.h"

using namespace PhysicsEngine;

Terrain::Terrain(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mMaterialId = Guid::INVALID;
    mCameraTransformId = Guid::INVALID;

    mCreated = false;
    mChanged = false;
    mMaterialChanged = false;
    mGrassMeshChanged = false;
    mTreeMeshChanged = false;

    mChunkSize = glm::vec2(10, 10);
    mChunkResolution = glm::ivec2(20, 20);

    mTotalChunkCount = 9;
    mGrassMeshCount = 0;
    mTreeMeshCount = 0;
    mMaxViewDistance = 10.0f;

    mScale = 1.0f;
    mAmplitude = 1.0f;
    mOffsetX = 1.0f;
    mOffsetZ = 1.0f;

    mVertexBuffer = VertexBuffer::create();
    mNormalBuffer = VertexBuffer::create();
    mTexCoordsBuffer = VertexBuffer::create();

    mHandle = MeshHandle::create();

    mHandle->addVertexBuffer(mVertexBuffer, AttribType::Vec3);
    mHandle->addVertexBuffer(mNormalBuffer, AttribType::Vec3);
    mHandle->addVertexBuffer(mTexCoordsBuffer, AttribType::Vec2);
}

Terrain::Terrain(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mMaterialId = Guid::INVALID;
    mCameraTransformId = Guid::INVALID;

    mCreated = false;
    mChanged = false;
    mMaterialChanged = false;
    mGrassMeshChanged = false;
    mTreeMeshChanged = false;

    mChunkSize = glm::vec2(10, 10);
    mChunkResolution = glm::ivec2(20, 20);

    mTotalChunkCount = 9;
    mGrassMeshCount = 0;
    mTreeMeshCount = 0;
    mMaxViewDistance = 10.0f;

    mScale = 1.0f;
    mAmplitude = 1.0f;
    mOffsetX = 1.0f;
    mOffsetZ = 1.0f;

    mVertexBuffer = VertexBuffer::create();
    mNormalBuffer = VertexBuffer::create();
    mTexCoordsBuffer = VertexBuffer::create();

    mHandle = MeshHandle::create();

    mHandle->addVertexBuffer(mVertexBuffer, AttribType::Vec3);
    mHandle->addVertexBuffer(mNormalBuffer, AttribType::Vec3);
    mHandle->addVertexBuffer(mTexCoordsBuffer, AttribType::Vec2);
}

Terrain::~Terrain()
{
    delete mVertexBuffer;
    delete mNormalBuffer;
    delete mTexCoordsBuffer;

    delete mHandle;
}

void Terrain::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["entityId"] = mEntityGuid;

    out["cameraTransformId"] = mCameraTransformId;
    out["materialId"] = mMaterialId;
    out["maxViewDistance"] = mMaxViewDistance;
    out["scale"] = mScale;
    out["amplitude"] = mAmplitude;
}

void Terrain::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mEntityGuid = YAML::getValue<Guid>(in, "entityId");

    mCameraTransformId = YAML::getValue<Guid>(in, "cameraTransformId");
    mMaterialId = YAML::getValue<Guid>(in, "materialId");
    mMaxViewDistance = YAML::getValue<float>(in, "maxViewDistance");
    mScale = YAML::getValue<float>(in, "scale");
    mAmplitude = YAML::getValue<float>(in, "amplitude");

    mMaterialChanged = true;
}

int Terrain::getType() const
{
    return PhysicsEngine::TERRAIN_TYPE;
}

std::string Terrain::getObjectName() const
{
    return PhysicsEngine::TERRAIN_NAME;
}

Guid Terrain::getEntityGuid() const
{
    return mEntityGuid;
}

Guid Terrain::getGuid() const
{
    return mGuid;
}

Id Terrain::getId() const
{
    return mId;
}

void Terrain::generateTerrain()
{
    if (mCreated)
    {
        return;
    }

    // determine number of chunks required based on view distance
    int chunkCountX = 3;
    int chunkCountZ = 3;
    if (mMaxViewDistance > 35.0f)
    {
        chunkCountX = 9;
        chunkCountZ = 9;
    }
    else if (mMaxViewDistance > 25.0f)
    {
        chunkCountX = 7;
        chunkCountZ = 7;
    }
    else if (mMaxViewDistance > 15.0f)
    {
        chunkCountX = 5;
        chunkCountZ = 5;
    }

    mTotalChunkCount = chunkCountX * chunkCountZ;

    // Generate plane mesh centered at (0, 0)
    int triangleCount = 2 * (mChunkResolution.x - 1) * (mChunkResolution.y - 1);
    int vertexCount = 3 * triangleCount;

    mPlaneVertices.resize(3 * vertexCount);
    mPlaneTexCoords.resize(2 * vertexCount);

    float xmin = -0.5f * mChunkSize.x;
    float xmax = 0.5f * mChunkSize.x;
    float zmin = -0.5f * mChunkSize.y;
    float zmax = 0.5f * mChunkSize.y;

    float dx = (xmax - xmin) / (mChunkResolution.x - 1);
    float dz = (zmax - zmin) / (mChunkResolution.y - 1);

    int ii = 0;
    int kk = 0;
    for (int z = 0; z < (mChunkResolution.y - 1); z++)
    {
        for (int x = 0; x < (mChunkResolution.x - 1); x++)
        {
            // first triangle (clockwise winding order)
            mPlaneVertices[ii++] = xmin + x * dx;
            mPlaneVertices[ii++] = 0.0f;
            mPlaneVertices[ii++] = zmin + z * dz;

            mPlaneVertices[ii++] = xmin + x * dx;
            mPlaneVertices[ii++] = 0.0f;
            mPlaneVertices[ii++] = zmin + (z + 1) * dz;

            mPlaneVertices[ii++] = xmin + (x + 1) * dx;
            mPlaneVertices[ii++] = 0.0f;
            mPlaneVertices[ii++] = zmin + z * dz;

            mPlaneTexCoords[kk++] = x * dx;
            mPlaneTexCoords[kk++] = z * dz;

            mPlaneTexCoords[kk++] = x * dx;
            mPlaneTexCoords[kk++] = (z + 1) * dz;

            mPlaneTexCoords[kk++] = (x + 1) * dx;
            mPlaneTexCoords[kk++] = z * dz;

            // second triangle (clockwise winding order)
            mPlaneVertices[ii++] = xmin + (x + 1) * dx;
            mPlaneVertices[ii++] = 0.0f;
            mPlaneVertices[ii++] = zmin + z * dz;

            mPlaneVertices[ii++] = xmin + x * dx;
            mPlaneVertices[ii++] = 0.0f;
            mPlaneVertices[ii++] = zmin + (z + 1) * dz;

            mPlaneVertices[ii++] = xmin + (x + 1) * dx;
            mPlaneVertices[ii++] = 0.0f;
            mPlaneVertices[ii++] = zmin + (z + 1) * dz;

            mPlaneTexCoords[kk++] = (x + 1) * dx;
            mPlaneTexCoords[kk++] = z * dz;

            mPlaneTexCoords[kk++] = x * dx;
            mPlaneTexCoords[kk++] = (z + 1) * dz;

            mPlaneTexCoords[kk++] = (x + 1) * dx;
            mPlaneTexCoords[kk++] = (z + 1) * dz;
        }
    }

    std::vector<glm::vec2> offsets(mTotalChunkCount);
    for (int z = 0; z < chunkCountZ; z++)
    {
        for (int x = 0; x < chunkCountX; x++)
        {
            offsets[chunkCountX * z + x] = glm::vec2(mChunkSize.x * x - mChunkSize.x, mChunkSize.y - mChunkSize.y * z);
        }
    }

    mVertices.resize(mTotalChunkCount * 3 * vertexCount);
    mNormals.resize(mTotalChunkCount * 3 * vertexCount);
    mTexCoords.resize(mTotalChunkCount * 2 * vertexCount);

    for (int i = 0; i < mTotalChunkCount; i++)
    {
        int chunk_start = (3 * vertexCount) * i;
        int chunk_end = (3 * vertexCount) * (i + 1);

        for (size_t j = 0; j < mPlaneVertices.size() / 3; j++)
        {
            float x = mPlaneVertices[3 * j + 0];
            float z = mPlaneVertices[3 * j + 2];

            x += offsets[i].x;
            z += offsets[i].y;

            mVertices[chunk_start + 3 * j + 0] = x;
            mVertices[chunk_start + 3 * j + 1] =
                mAmplitude * stb_perlin_noise3(mScale * x + mOffsetX, 0, mScale * z + mOffsetZ, 0, 0, 0);
            mVertices[chunk_start + 3 * j + 2] = z;
        }

        for (size_t j = 0; j < mPlaneTexCoords.size(); j++)
        {
            mTexCoords[(2 * vertexCount) * i + j] = mPlaneTexCoords[j];
        }

        mTerrainChunks[i].mStart = chunk_start;
        mTerrainChunks[i].mEnd = chunk_end;
        mTerrainChunks[i].mEnabled = true;

        float rectWidth = mChunkSize.x;
        float rectHeight = mChunkSize.y;
        glm::vec2 rectCentre = glm::vec2(offsets[i].x, offsets[i].y);

        float rectX = rectCentre.x - 0.5f * rectWidth;
        float rectY = rectCentre.y - 0.5f * rectHeight;

        mTerrainChunks[i].mRect = Rect(rectX, rectY, mChunkSize.x, mChunkSize.y);
    }

    // Compute normals
    for (size_t j = 0; j < mVertices.size() / 9; j++)
    {
        // vertex 1
        float vx1 = mVertices[9 * j + 0];
        float vy1 = mVertices[9 * j + 1];
        float vz1 = mVertices[9 * j + 2];

        // vertex 2
        float vx2 = mVertices[9 * j + 3];
        float vy2 = mVertices[9 * j + 4];
        float vz2 = mVertices[9 * j + 5];

        // vertex 3
        float vx3 = mVertices[9 * j + 6];
        float vy3 = mVertices[9 * j + 7];
        float vz3 = mVertices[9 * j + 8];

        // Calculate p vector
        float px = vx2 - vx1;
        float py = vy2 - vy1;
        float pz = vz2 - vz1;

        // Calculate q vector
        float qx = vx3 - vx1;
        float qy = vy3 - vy1;
        float qz = vz3 - vz1;

        // Calculate normal (p x q)
        // i  j  k
        // px py pz
        // qx qy qz
        float nx = py * qz - pz * qy;
        float ny = pz * qx - px * qz;
        float nz = px * qy - py * qx;
        // Scale to unit vector
        float s = sqrt(nx * nx + ny * ny + nz * nz);
        nx /= s;
        ny /= s;
        nz /= s;

        // Add the normal 3 times (once for each vertex)
        mNormals[9 * j + 0] = nx;
        mNormals[9 * j + 1] = ny;
        mNormals[9 * j + 2] = nz;

        mNormals[9 * j + 3] = nx;
        mNormals[9 * j + 4] = ny;
        mNormals[9 * j + 5] = nz;

        mNormals[9 * j + 6] = nx;
        mNormals[9 * j + 7] = ny;
        mNormals[9 * j + 8] = nz;
    }

    mVertexBuffer->bind();
    if (mVertexBuffer->getSize() < sizeof(float) * mVertices.size())
    {
        mVertexBuffer->resize(sizeof(float) * mVertices.size());
    }
    mVertexBuffer->setData(mVertices.data(), 0, sizeof(float) * mVertices.size());
    mVertexBuffer->unbind();

    mNormalBuffer->bind();
    if (mNormalBuffer->getSize() < sizeof(float) * mNormals.size())
    {
        mNormalBuffer->resize(sizeof(float) * mNormals.size());
    }
    mNormalBuffer->setData(mNormals.data(), 0, sizeof(float) * mNormals.size());
    mNormalBuffer->unbind();

    mTexCoordsBuffer->bind();
    if (mTexCoordsBuffer->getSize() < sizeof(float) * mTexCoords.size())
    {
        mTexCoordsBuffer->resize(sizeof(float) * mTexCoords.size());
    }
    mTexCoordsBuffer->setData(mTexCoords.data(), 0, sizeof(float) * mTexCoords.size());
    mTexCoordsBuffer->unbind();

    mCreated = true;
}

void Terrain::regenerateTerrain()
{
    // determine number of chunks required based on view distance
    int chunkCountX = 3;
    int chunkCountZ = 3;
    if (mMaxViewDistance > 35.0f)
    {
        chunkCountX = 9;
        chunkCountZ = 9;
    }
    else if (mMaxViewDistance > 25.0f)
    {
        chunkCountX = 7;
        chunkCountZ = 7;
    }
    else if (mMaxViewDistance > 15.0f)
    {
        chunkCountX = 5;
        chunkCountZ = 5;
    }

    mTotalChunkCount = chunkCountX * chunkCountZ;

    // Generate plane mesh centered at (0, 0)
    int triangleCount = 2 * (mChunkResolution.x - 1) * (mChunkResolution.y - 1);
    int vertexCount = 3 * triangleCount;

    std::vector<glm::vec2> offsets(mTotalChunkCount);
    for (int z = 0; z < chunkCountZ; z++)
    {
        for (int x = 0; x < chunkCountX; x++)
        {
            offsets[chunkCountX * z + x] = glm::vec2(mChunkSize.x * x - mChunkSize.x, mChunkSize.y - mChunkSize.y * z);
        }
    }

    mVertices.resize(mTotalChunkCount * 3 * vertexCount);
    mNormals.resize(mTotalChunkCount * 3 * vertexCount);
    mTexCoords.resize(mTotalChunkCount * 2 * vertexCount);

    for (int i = 0; i < mTotalChunkCount; i++)
    {
        int chunk_start = (3 * vertexCount) * i;
        int chunk_end = (3 * vertexCount) * (i + 1);

        for (size_t j = 0; j < mPlaneVertices.size() / 3; j++)
        {
            float x = mPlaneVertices[3 * j + 0];
            float z = mPlaneVertices[3 * j + 2];

            x += offsets[i].x;
            z += offsets[i].y;

            mVertices[chunk_start + 3 * j + 0] = x;
            mVertices[chunk_start + 3 * j + 1] =
                mAmplitude * stb_perlin_noise3(mScale * x + mOffsetX, 0, mScale * z + mOffsetZ, 0, 0, 0);
            mVertices[chunk_start + 3 * j + 2] = z;
        }

        for (size_t j = 0; j < mPlaneTexCoords.size(); j++)
        {
            mTexCoords[(2 * vertexCount) * i + j] = mPlaneTexCoords[j];
        }

        mTerrainChunks[i].mStart = chunk_start;
        mTerrainChunks[i].mEnd = chunk_end;
        mTerrainChunks[i].mEnabled = true;

        float rectWidth = mChunkSize.x;
        float rectHeight = mChunkSize.y;
        glm::vec2 rectCentre = glm::vec2(offsets[i].x, offsets[i].y);

        float rectX = rectCentre.x - 0.5f * rectWidth;
        float rectY = rectCentre.y - 0.5f * rectHeight;

        mTerrainChunks[i].mRect = Rect(rectX, rectY, mChunkSize.x, mChunkSize.y);
    }

    // Compute normals
    for (size_t j = 0; j < mVertices.size() / 9; j++)
    {
        // vertex 1
        float vx1 = mVertices[9 * j + 0];
        float vy1 = mVertices[9 * j + 1];
        float vz1 = mVertices[9 * j + 2];

        // vertex 2
        float vx2 = mVertices[9 * j + 3];
        float vy2 = mVertices[9 * j + 4];
        float vz2 = mVertices[9 * j + 5];

        // vertex 3
        float vx3 = mVertices[9 * j + 6];
        float vy3 = mVertices[9 * j + 7];
        float vz3 = mVertices[9 * j + 8];

        // Calculate p vector
        float px = vx2 - vx1;
        float py = vy2 - vy1;
        float pz = vz2 - vz1;

        // Calculate q vector
        float qx = vx3 - vx1;
        float qy = vy3 - vy1;
        float qz = vz3 - vz1;

        // Calculate normal (p x q)
        // i  j  k
        // px py pz
        // qx qy qz
        float nx = py * qz - pz * qy;
        float ny = pz * qx - px * qz;
        float nz = px * qy - py * qx;
        // Scale to unit vector
        float s = sqrt(nx * nx + ny * ny + nz * nz);
        nx /= s;
        ny /= s;
        nz /= s;

        // Add the normal 3 times (once for each vertex)
        mNormals[9 * j + 0] = nx;
        mNormals[9 * j + 1] = ny;
        mNormals[9 * j + 2] = nz;

        mNormals[9 * j + 3] = nx;
        mNormals[9 * j + 4] = ny;
        mNormals[9 * j + 5] = nz;

        mNormals[9 * j + 6] = nx;
        mNormals[9 * j + 7] = ny;
        mNormals[9 * j + 8] = nz;
    }

    mVertexBuffer->bind();
    if (mVertexBuffer->getSize() < sizeof(float) * mVertices.size())
    {
        mVertexBuffer->resize(sizeof(float) * mVertices.size());
    }
    mVertexBuffer->setData(mVertices.data(), 0, sizeof(float) * mVertices.size());
    mVertexBuffer->unbind();

    mNormalBuffer->bind();
    if (mNormalBuffer->getSize() < sizeof(float) * mNormals.size())
    {
        mNormalBuffer->resize(sizeof(float) * mNormals.size());
    }
    mNormalBuffer->setData(mNormals.data(), 0, sizeof(float) * mNormals.size());
    mNormalBuffer->unbind();

    mTexCoordsBuffer->bind();
    if (mTexCoordsBuffer->getSize() < sizeof(float) * mTexCoords.size())
    {
        mTexCoordsBuffer->resize(sizeof(float) * mTexCoords.size());
    }
    mTexCoordsBuffer->setData(mTexCoords.data(), 0, sizeof(float) * mTexCoords.size());
    mTexCoordsBuffer->unbind();
}

void Terrain::updateTerrainHeight(float dx, float dz)
{
    for (size_t j = 0; j < mVertices.size() / 3; j++)
    {
        float x = mVertices[3 * j + 0] + dx;
        float z = mVertices[3 * j + 2] + dz;

        mVertices[3 * j + 0] = x;
        mVertices[3 * j + 1] = mAmplitude * stb_perlin_noise3(mScale * x + mOffsetX, 0, mScale * z + mOffsetZ, 0, 0, 0);
        mVertices[3 * j + 2] = z;
    }

    for (int i = 0; i < mTotalChunkCount; i++)
    {
        float rectWidth = mChunkSize.x;
        float rectHeight = mChunkSize.y;
        glm::vec2 rectCentre = mTerrainChunks[i].mRect.getCentre();

        rectCentre.x += dx;
        rectCentre.y += dz;

        float rectX = rectCentre.x - 0.5f * rectWidth;
        float rectY = rectCentre.y - 0.5f * rectHeight;

        mTerrainChunks[i].mRect = Rect(rectX, rectY, mChunkSize.x, mChunkSize.y);
    }

    for (size_t j = 0; j < mVertices.size() / 9; j++)
    {
        // vertex 1
        float vx1 = mVertices[9 * j + 0];
        float vy1 = mVertices[9 * j + 1];
        float vz1 = mVertices[9 * j + 2];

        // vertex 2
        float vx2 = mVertices[9 * j + 3];
        float vy2 = mVertices[9 * j + 4];
        float vz2 = mVertices[9 * j + 5];

        // vertex 3
        float vx3 = mVertices[9 * j + 6];
        float vy3 = mVertices[9 * j + 7];
        float vz3 = mVertices[9 * j + 8];

        // Calculate p vector
        float px = vx2 - vx1;
        float py = vy2 - vy1;
        float pz = vz2 - vz1;
        // Calculate q vector
        float qx = vx3 - vx1;
        float qy = vy3 - vy1;
        float qz = vz3 - vz1;

        // Calculate normal (p x q)
        // i  j  k
        // px py pz
        // qx qy qz
        float nx = py * qz - pz * qy;
        float ny = pz * qx - px * qz;
        float nz = px * qy - py * qx;
        // Scale to unit vector
        float s = sqrt(nx * nx + ny * ny + nz * nz);
        nx /= s;
        ny /= s;
        nz /= s;

        // Add the normal 3 times (once for each vertex)
        mNormals[9 * j + 0] = nx;
        mNormals[9 * j + 1] = ny;
        mNormals[9 * j + 2] = nz;

        mNormals[9 * j + 3] = nx;
        mNormals[9 * j + 4] = ny;
        mNormals[9 * j + 5] = nz;

        mNormals[9 * j + 6] = nx;
        mNormals[9 * j + 7] = ny;
        mNormals[9 * j + 8] = nz;
    }

    mVertexBuffer->bind();
    if (mVertexBuffer->getSize() < sizeof(float) * mVertices.size())
    {
        mVertexBuffer->resize(sizeof(float) * mVertices.size());
    }
    mVertexBuffer->setData(mVertices.data(), 0, sizeof(float) * mVertices.size());
    mVertexBuffer->unbind();

    mNormalBuffer->bind();
    if (mNormalBuffer->getSize() < sizeof(float) * mNormals.size())
    {
        mNormalBuffer->resize(sizeof(float) * mNormals.size());
    }
    mNormalBuffer->setData(mNormals.data(), 0, sizeof(float) * mNormals.size());
    mNormalBuffer->unbind();
}

std::vector<float> Terrain::getVertices() const
{
    return mVertices;
}

std::vector<float> Terrain::getNormals() const
{
    return mNormals;
}

std::vector<float> Terrain::getTexCoords() const
{
    return mTexCoords;
}

MeshHandle *Terrain::getNativeGraphicsHandle() const
{
    return mHandle;
}

void Terrain::setMaterial(Guid materialId)
{
    mMaterialId = materialId;
    mMaterialChanged = true;
}

void Terrain::setGrassMesh(Guid meshId, int index)
{
    if (index >= 0 && index < 8)
    {
        mGrassMeshes[index].mMeshId = meshId;
        mGrassMeshChanged = true;
    }
}

void Terrain::setTreeMesh(Guid meshId, int index)
{
    if (index >= 0 && index < 8)
    {
        mTreeMeshes[index].mMeshId = meshId;
        mTreeMeshChanged = true;
    }
}

void Terrain::setGrassMaterial(Guid materialId, int meshIndex, int materialIndex)
{
    if (meshIndex >= 0 && meshIndex < 8)
    {
        if (materialIndex >= 0 && materialIndex < 8)
        {
            mGrassMeshes[meshIndex].mMaterialIds[materialIndex] = materialId;
            mGrassMeshChanged = true;
        }
    }
}

void Terrain::setTreeMaterial(Guid materialId, int meshIndex, int materialIndex)
{
    if (meshIndex >= 0 && meshIndex < 8)
    {
        if (materialIndex >= 0 && materialIndex < 8)
        {
            mTreeMeshes[meshIndex].mMaterialIds[materialIndex] = materialId;
            mTreeMeshChanged = true;
        }
    }
}

Guid Terrain::getMaterial() const
{
    return mMaterialId;
}

Guid Terrain::getGrassMesh(int index) const
{
    if (index >= 0 && index < 8)
    {
        return mGrassMeshes[index].mMeshId;
    }

    return Guid::INVALID;
}

Guid Terrain::getTreeMesh(int index) const
{
    if (index >= 0 && index < 8)
    {
        return mTreeMeshes[index].mMeshId;
    }

    return Guid::INVALID;
}

Guid Terrain::getGrassMesh(int meshIndex, int materialIndex) const
{
    if (meshIndex >= 0 && meshIndex < 8)
    {
        if (materialIndex >= 0 && materialIndex < 8)
        {
            return mGrassMeshes[meshIndex].mMaterialIds[materialIndex];
        }
    }

    return Guid::INVALID;
}

Guid Terrain::getTreeMesh(int meshIndex, int materialIndex) const
{
    if (meshIndex >= 0 && meshIndex < 8)
    {
        if (materialIndex >= 0 && materialIndex < 8)
        {
            return mTreeMeshes[meshIndex].mMaterialIds[materialIndex];
        }
    }

    return Guid::INVALID;
}

bool Terrain::isCreated() const
{
    return mCreated;
}

bool Terrain::isChunkEnabled(int chunk) const
{
    assert(chunk >= 0);
    assert(chunk < 81);
    return mTerrainChunks[chunk].mEnabled;
}

void Terrain::enableChunk(int chunk)
{
    assert(chunk >= 0);
    assert(chunk < 81);
    mTerrainChunks[chunk].mEnabled = true;
}

void Terrain::disableChunk(int chunk)
{
    assert(chunk >= 0);
    assert(chunk < 81);
    mTerrainChunks[chunk].mEnabled = false;
}

size_t Terrain::getChunkStart(int chunk) const
{
    assert(chunk >= 0);
    assert(chunk < 81);
    return mTerrainChunks[chunk].mStart;
}

size_t Terrain::getChunkSize(int chunk) const
{
    assert(chunk >= 0);
    assert(chunk < 81);
    return mTerrainChunks[chunk].mEnd - mTerrainChunks[chunk].mStart;
}

Sphere Terrain::getChunkBounds(int chunk) const
{
    glm::vec2 centre = mTerrainChunks[chunk].mRect.getCentre();
    float radius = glm::sqrt(mChunkSize.x * mChunkSize.x + mChunkSize.y * mChunkSize.y);
    return Sphere(glm::vec3(centre.x, 0, centre.y), radius);
}

Rect Terrain::getChunkRect(int chunk) const
{
    assert(chunk >= 0);
    assert(chunk < 81);
    return mTerrainChunks[chunk].mRect;
}

Rect Terrain::getCentreChunkRect() const
{
    return mTerrainChunks[(mTotalChunkCount - 1) / 2].mRect;
}

int Terrain::getTotalChunkCount() const
{
    return mTotalChunkCount;
}