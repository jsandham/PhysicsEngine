#include "../../include/core/Mesh.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"
#include "../../include/obj_load/obj_load.h"

using namespace PhysicsEngine;

Mesh::Mesh()
{
    mAssetId = Guid::INVALID;
    mCreated = false;
}

Mesh::Mesh(const std::vector<char> &data)
{
    deserialize(data);
}

Mesh::~Mesh()
{
}

std::vector<char> Mesh::serialize() const
{
    return serialize(mAssetId);
}

std::vector<char> Mesh::serialize(Guid assetId) const
{
    MeshHeader header;
    header.mMeshId = assetId;
    header.mVerticesSize = mVertices.size();
    header.mNormalsSize = mNormals.size();
    header.mTexCoordsSize = mTexCoords.size();
    header.mSubMeshVertexStartIndiciesSize = mSubMeshVertexStartIndices.size();

    size_t numberOfBytes = sizeof(MeshHeader) + mVertices.size() * sizeof(float) + mNormals.size() * sizeof(float) +
                           mTexCoords.size() * sizeof(float) + mSubMeshVertexStartIndices.size() * sizeof(int);

    std::vector<char> data(numberOfBytes);

    size_t start1 = 0;
    size_t start2 = start1 + sizeof(MeshHeader);
    size_t start3 = start2 + sizeof(float) * mVertices.size();
    size_t start4 = start3 + sizeof(float) * mNormals.size();
    size_t start5 = start4 + sizeof(float) * mTexCoords.size();

    memcpy(&data[start1], &header, sizeof(MeshHeader));
    memcpy(&data[start2], &mVertices[0], sizeof(float) * mVertices.size());
    memcpy(&data[start3], &mNormals[0], sizeof(float) * mNormals.size());
    memcpy(&data[start4], &mTexCoords[0], sizeof(float) * mTexCoords.size());
    memcpy(&data[start5], &mSubMeshVertexStartIndices[0], sizeof(int) * mSubMeshVertexStartIndices.size());

    return data;
}

void Mesh::deserialize(const std::vector<char> &data)
{
    size_t start1 = 0;
    size_t start2 = start1 + sizeof(MeshHeader);

    const MeshHeader *header = reinterpret_cast<const MeshHeader *>(&data[start1]);

    mAssetId = header->mMeshId;
    mVertices.resize(header->mVerticesSize);
    mNormals.resize(header->mNormalsSize);
    mTexCoords.resize(header->mTexCoordsSize);
    mSubMeshVertexStartIndices.resize(header->mSubMeshVertexStartIndiciesSize);

    size_t start3 = start2 + sizeof(float) * mVertices.size();
    size_t start4 = start3 + sizeof(float) * mNormals.size();
    size_t start5 = start4 + sizeof(float) * mTexCoords.size();

    for (size_t i = 0; i < header->mVerticesSize; i++)
    {
        mVertices[i] = *reinterpret_cast<const float *>(&data[start2 + sizeof(float) * i]);
    }

    for (size_t i = 0; i < header->mNormalsSize; i++)
    {
        mNormals[i] = *reinterpret_cast<const float *>(&data[start3 + sizeof(float) * i]);
    }

    for (size_t i = 0; i < header->mTexCoordsSize; i++)
    {
        mTexCoords[i] = *reinterpret_cast<const float *>(&data[start4 + sizeof(float) * i]);
    }

    for (size_t i = 0; i < header->mSubMeshVertexStartIndiciesSize; i++)
    {
        mSubMeshVertexStartIndices[i] = *reinterpret_cast<const int *>(&data[start5 + sizeof(int) * i]);
    }

    mCreated = false;
}

void Mesh::load(const std::string &filepath)
{
    obj_mesh mesh;

    if (obj_load(filepath, mesh))
    {
        mVertices = mesh.mVertices;
        mNormals = mesh.mNormals;
        mTexCoords = mesh.mTexCoords;
        mSubMeshVertexStartIndices = mesh.mSubMeshVertexStartIndices;

        mCreated = false;
    }
    else
    {
        std::string message = "Error: Could not load obj mesh " + filepath + "\n";
        Log::error(message.c_str());
    }
}

void Mesh::load(std::vector<float> vertices, std::vector<float> normals, std::vector<float> texCoords,
                std::vector<int> subMeshStartIndices)
{
    mVertices = vertices;
    mNormals = normals;
    mTexCoords = texCoords;
    mSubMeshVertexStartIndices = subMeshStartIndices;

    mCreated = false;
}

Sphere Mesh::computeBoundingSphere() const
{
    Sphere sphere;
    sphere.mRadius = 1.0f;
    sphere.mCentre = glm::vec3(0.0f, 0.0f, 0.0f);

    size_t numVertices = mVertices.size() / 3;

    if (numVertices == 0)
    {
        std::string message = "Error: No vertices assigned in mesh when trying to compute bounding sphere\n";
        Log::error(message.c_str());
        return sphere;
    }

    // Ritter algorithm for bounding sphere
    // find furthest point from first vertex
    glm::vec3 x = glm::vec3(mVertices[0], mVertices[1], mVertices[2]);

    glm::vec3 y = x;
    float maxDistance = 0.0f;
    for (size_t i = 1; i < numVertices; i++)
    {

        glm::vec3 temp = glm::vec3(mVertices[3 * i], mVertices[3 * i + 1], mVertices[3 * i + 2]);
        float distance = glm::distance(x, temp);
        if (distance > maxDistance)
        {
            y = temp;
            maxDistance = distance;
        }
    }

    // now find furthest point from y
    glm::vec3 z = y;
    maxDistance = 0.0f;
    for (size_t i = 0; i < numVertices; i++)
    {

        glm::vec3 temp = glm::vec3(mVertices[3 * i], mVertices[3 * i + 1], mVertices[3 * i + 2]);
        float distance = glm::distance(y, temp);
        if (distance > maxDistance)
        {
            z = temp;
            maxDistance = distance;
        }
    }

    sphere.mRadius = 0.5f * glm::distance(y, z);
    sphere.mCentre = 0.5f * (y + z);

    for (size_t i = 0; i < numVertices; i++)
    {
        glm::vec3 temp = glm::vec3(mVertices[3 * i], mVertices[3 * i + 1], mVertices[3 * i + 2]);
        float radius = glm::distance(temp, sphere.mCentre);
        if (radius > sphere.mRadius)
        {
            sphere.mRadius = radius;
        }
    }

    return sphere;
}

bool Mesh::isCreated() const
{
    return mCreated;
}

const std::vector<float> &Mesh::getVertices() const
{
    return mVertices;
}

const std::vector<float> &Mesh::getNormals() const
{
    return mNormals;
}

const std::vector<float> &Mesh::getTexCoords() const
{
    return mTexCoords;
}

const std::vector<int> &Mesh::getSubMeshStartIndices() const
{
    return mSubMeshVertexStartIndices;
}

int Mesh::getSubMeshStartIndex(int subMeshIndex) const
{
    if (subMeshIndex >= mSubMeshVertexStartIndices.size() - 1)
    {
        return -1;
    }

    return mSubMeshVertexStartIndices[subMeshIndex];
}

int Mesh::getSubMeshEndIndex(int subMeshIndex) const
{
    if (subMeshIndex >= mSubMeshVertexStartIndices.size() - 1)
    {
        return -1;
    }

    return mSubMeshVertexStartIndices[subMeshIndex + 1];
}

int Mesh::getSubMeshCount() const
{
    return (int)mSubMeshVertexStartIndices.size() - 1;
}

GLuint Mesh::getNativeGraphicsVAO() const
{
    return mVao;
}

void Mesh::create()
{
    if (mCreated)
    {
        return;
    }

    Graphics::createMesh(mVertices, mNormals, mTexCoords, &mVao, &mVbo[0], &mVbo[1], &mVbo[2]);

    mCreated = true;
}

void Mesh::destroy()
{
    if (!mCreated)
    {
        return;
    }

    Graphics::destroyMesh(&mVao, &mVbo[0], &mVbo[1], &mVbo[2]);

    mCreated = false;
}

void Mesh::writeMesh()
{
    
}