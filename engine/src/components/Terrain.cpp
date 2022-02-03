#include "../../include/components/Terrain.h"
#include "../../include/graphics/Graphics.h"

#include "stb_perlin.h"

#include <array>

using namespace PhysicsEngine;

Terrain::Terrain(World *world) : Component(world)
{
    mMaterialId = Guid::INVALID;
    mCameraTransformId = Guid::INVALID;

    mCreated = false;
    mChanged = false;
    mMaterialChanged = false;

    mScale = 1.0f;
    mAmplitude = 1.0f;
    mOffsetX = 1.0f;
    mOffsetZ = 1.0f;
}

Terrain::Terrain(World *world, const Guid &id) : Component(world, id)
{
    mMaterialId = Guid::INVALID;
    mCameraTransformId = Guid::INVALID;

    mCreated = false;
    mChanged = false;
    mMaterialChanged = false;

    mScale = 1.0f;
    mAmplitude = 1.0f;
    mOffsetX = 1.0f;
    mOffsetZ = 1.0f;
}

Terrain::~Terrain()
{

}

void Terrain::serialize(YAML::Node& out) const
{
    Component::serialize(out);

    out["cameraTransformId"] = mCameraTransformId;
    out["materialId"] = mMaterialId;
    out["scale"] = mScale;
    out["amplitude"] = mAmplitude;

}

void Terrain::deserialize(const YAML::Node& in)
{
    Component::deserialize(in);

    mCameraTransformId = YAML::getValue<Guid>(in, "cameraTransformId");
    mMaterialId = YAML::getValue<Guid>(in, "materialId");
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

void Terrain::generateTerrain()
{
    if (mCreated)
    {
        return;
    }

    int nx = 100;
    int nz = 100;

    mChunkSize = glm::ivec2(10, 10);

    // Generate plane mesh centered at (0, 0)
    int triangleCount = 2 * (nx - 1) * (nz - 1);
    int vertexCount = 3 * triangleCount;

    std::vector<float> vertices(3 * vertexCount);
    std::vector<float> texCoords(2 * vertexCount);

    float xmin = -0.5f * mChunkSize.x;
    float xmax = 0.5f * mChunkSize.x;
    float zmin = -0.5f * mChunkSize.y;
    float zmax = 0.5f * mChunkSize.y;

    float dx = (xmax - xmin) / (nx - 1);
    float dz = (zmax - zmin) / (nz - 1);

    int ii = 0;
    int kk = 0;
    for (int z = 0; z < (nz - 1); z++)
    {
        for (int x = 0; x < (nx - 1); x++)
        {
            // first triangle (clockwise winding order)
            vertices[ii++] = xmin + x * dx;
            vertices[ii++] = 0.0f;
            vertices[ii++] = zmin + z * dz;

            vertices[ii++] = xmin + x * dx;
            vertices[ii++] = 0.0f;
            vertices[ii++] = zmin + (z + 1) * dz;

            vertices[ii++] = xmin + (x + 1) * dx;
            vertices[ii++] = 0.0f;
            vertices[ii++] = zmin + z * dz;

            texCoords[kk++] = x * dx;
            texCoords[kk++] = z * dz;

            texCoords[kk++] = x * dx;
            texCoords[kk++] = (z + 1) * dz;

            texCoords[kk++] = (x + 1) * dx;
            texCoords[kk++] = z * dz;

            // second triangle (clockwise winding order)
            vertices[ii++] = xmin + (x + 1) * dx;
            vertices[ii++] = 0.0f;
            vertices[ii++] = zmin + z * dz;

            vertices[ii++] = xmin + x * dx;
            vertices[ii++] = 0.0f;
            vertices[ii++] = zmin + (z + 1) * dz;

            vertices[ii++] = xmin + (x + 1) * dx;
            vertices[ii++] = 0.0f;
            vertices[ii++] = zmin + (z + 1) * dz;

            texCoords[kk++] = (x + 1) * dx;
            texCoords[kk++] = z * dz;

            texCoords[kk++] = x * dx;
            texCoords[kk++] = (z + 1) * dz;

            texCoords[kk++] = (x + 1) * dx;
            texCoords[kk++] = (z + 1) * dz;
        }
    }

    std::array<glm::ivec2, 9> offsets = {glm::ivec2(-mChunkSize.x, mChunkSize.y),
                                         glm::ivec2(0, mChunkSize.y),
                                         glm::ivec2(mChunkSize.x, mChunkSize.y),
                                         glm::ivec2(-mChunkSize.x, 0),
                                         glm::ivec2(0, 0),
                                         glm::ivec2(mChunkSize.x, 0),
                                         glm::ivec2(-mChunkSize.x, -mChunkSize.y),
                                         glm::ivec2(0, -mChunkSize.y),
                                         glm::ivec2(mChunkSize.x, -mChunkSize.y)};

    mVertices.resize(9 * 3 * vertexCount);
    mNormals.resize(9 * 3 * vertexCount);
    mTexCoords.resize(9 * 2 * vertexCount);

    for (int i = 0; i < 9; i++)
    {
        int chunk_start = (3 * vertexCount) * i;
        int chunk_end = (3 * vertexCount) * (i + 1);

        for (size_t j = 0; j < vertices.size() / 3; j++)
        {
            float x = vertices[3 * j + 0];
            float z = vertices[3 * j + 2];

            x += offsets[i].x;
            z += offsets[i].y;

            mVertices[chunk_start + 3 * j + 0] = x;
            mVertices[chunk_start + 3 * j + 1] =
                mAmplitude * stb_perlin_noise3(mScale * x + mOffsetX, 0, mScale * z + mOffsetZ, 0, 0, 0);
            mVertices[chunk_start + 3 * j + 2] = z;
        }

        for (size_t j = 0; j < texCoords.size(); j++)
        {
            mTexCoords[(2 * vertexCount) * i + j] = texCoords[j];
        }

        mTerrainChunks[i].mStart = chunk_start;
        mTerrainChunks[i].mEnd = chunk_end;
        mTerrainChunks[i].mEnabled = true;
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

    Graphics::createTerrainChunk(mVertices, mNormals, mTexCoords,
                                 &mVao, &mVbo[0], &mVbo[1], &mVbo[2]);

    mCreated = true;
}

void Terrain::regenerateTerrain()
{
    for (size_t j = 0; j < mVertices.size() / 3; j++)
    {
        float x = mVertices[3 * j + 0];
        float z = mVertices[3 * j + 2];

        mVertices[3 * j + 1] =
            mAmplitude * stb_perlin_noise3(mScale * x + mOffsetX, 0, mScale * z + mOffsetZ, 0, 0, 0);
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

    Graphics::updateTerrainChunk(mVertices, mNormals, mVbo[0], mVbo[1]);
}

void Terrain::refine(int level)
{

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

unsigned int Terrain::getNativeGraphicsVAO() const
{
    return mVao;
}

void Terrain::setMaterial(Guid materialId)
{
    mMaterialId = materialId;
    mMaterialChanged = true;
}

Guid Terrain::getMaterial() const
{
    return mMaterialId;
}

bool Terrain::isCreated() const
{
    return mCreated;
}

bool Terrain::isChunkEnabled(int chunk) const
{
    assert(chunk >= 0);
    assert(chunk < 9);
    return mTerrainChunks[chunk].mEnabled;
}

void Terrain::enableChunk(int chunk)
{
    assert(chunk >= 0);
    assert(chunk < 9);
    mTerrainChunks[chunk].mEnabled = true;
}

void Terrain::disableChunk(int chunk)
{
    assert(chunk >= 0);
    assert(chunk < 9);
    mTerrainChunks[chunk].mEnabled = false;
}

size_t Terrain::getChunkStart(int chunk) const
{
    assert(chunk >= 0);
    assert(chunk < 9);
    return mTerrainChunks[chunk].mStart;
}

size_t Terrain::getChunkSize(int chunk) const
{
    assert(chunk >= 0);
    assert(chunk < 9);
    return mTerrainChunks[chunk].mEnd - mTerrainChunks[chunk].mStart;
}