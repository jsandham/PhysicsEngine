#include "../../include/core/Mesh.h"
#include "../../include/core/Log.h"
#include "../../include/core/Serialization.h"
#include "../../include/graphics/Graphics.h"
#include "../../include/obj_load/obj_load.h"

using namespace PhysicsEngine;

Mesh::Mesh() : Asset()
{
    mCreated = false;
    mChanged = false;
}

Mesh::Mesh(Guid id) : Asset(id)
{
    mCreated = false;
    mChanged = false;
}

Mesh::~Mesh()
{
}

void Mesh::serialize(std::ostream &out) const
{
    Asset::serialize(out);

    PhysicsEngine::write<size_t>(out, mVertices.size());
    PhysicsEngine::write<size_t>(out, mNormals.size());
    PhysicsEngine::write<size_t>(out, mTexCoords.size());
    PhysicsEngine::write<size_t>(out, mSubMeshVertexStartIndices.size());
    PhysicsEngine::write<const float>(out, mVertices.data(), mVertices.size());
    PhysicsEngine::write<const float>(out, mNormals.data(), mNormals.size());
    PhysicsEngine::write<const float>(out, mTexCoords.data(), mTexCoords.size());
    PhysicsEngine::write<const int>(out, mSubMeshVertexStartIndices.data(), mSubMeshVertexStartIndices.size());
}

void Mesh::deserialize(std::istream &in)
{
    Asset::deserialize(in);

    size_t vertexCount, normalCount, texCoordCount, subMeshCount;
    PhysicsEngine::read<size_t>(in, vertexCount);
    PhysicsEngine::read<size_t>(in, normalCount);
    PhysicsEngine::read<size_t>(in, texCoordCount);
    PhysicsEngine::read<size_t>(in, subMeshCount);

    mVertices.resize(vertexCount);
    mNormals.resize(normalCount);
    mTexCoords.resize(texCoordCount);
    mSubMeshVertexStartIndices.resize(subMeshCount);

    PhysicsEngine::read<float>(in, mVertices.data(), vertexCount);
    PhysicsEngine::read<float>(in, mNormals.data(), normalCount);
    PhysicsEngine::read<float>(in, mTexCoords.data(), texCoordCount);
    PhysicsEngine::read<int>(in, mSubMeshVertexStartIndices.data(), subMeshCount);

    computeBoundingSphere();

    mCreated = false;
    mChanged = false;
}

void Mesh::serialize(YAML::Node& out) const
{
    Asset::serialize(out);

    out["source"] = "";
}

void Mesh::deserialize(const YAML::Node& in)
{
    Asset::deserialize(in);

    std::string source = in["source"].as<std::string>();
    load(source);
}

int Mesh::getType() const
{
    return PhysicsEngine::MESH_TYPE;
}

std::string Mesh::getObjectName() const
{
    return PhysicsEngine::MESH_NAME;
}

void Mesh::load(const std::string &filepath)
{
    if (filepath.empty()) {
        return;
    }

    obj_mesh mesh;

    if (obj_load(filepath, mesh))
    {
        mVertices = mesh.mVertices;
        mNormals = mesh.mNormals;
        mTexCoords = mesh.mTexCoords;
        mSubMeshVertexStartIndices = mesh.mSubMeshVertexStartIndices;

        computeBoundingSphere();

        mCreated = false;
    }
    else
    {
        Log::error(("Could not load obj mesh " + filepath + "\n").c_str());
    }
}

void Mesh::load(std::vector<float> vertices, std::vector<float> normals, std::vector<float> texCoords,
                std::vector<int> subMeshStartIndices)
{
    mVertices = vertices;
    mNormals = normals;
    mTexCoords = texCoords;
    mSubMeshVertexStartIndices = subMeshStartIndices;

    computeBoundingSphere();

    mCreated = false;
}

bool Mesh::isCreated() const
{
    return mCreated;
}

bool Mesh::isChanged() const
{
    return mChanged;
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

Sphere Mesh::getBounds() const
{
    return mBounds;
}

GLuint Mesh::getNativeGraphicsVAO() const
{
    return mVao;
}

void Mesh::setVertices(const std::vector<float> &vertices)
{
    mVertices = vertices;
    computeBoundingSphere();

    mChanged = true;
}

void Mesh::setNormals(const std::vector<float> &normals)
{
    mNormals = normals;

    mChanged = true;
}

void Mesh::setTexCoords(const std::vector<float> &texCoords)
{
    mTexCoords = texCoords;

    mChanged = true;
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

void Mesh::computeBoundingSphere()
{
    mBounds.mRadius = 0.0f;
    mBounds.mCentre = glm::vec3(0.0f, 0.0f, 0.0f);

    size_t numVertices = mVertices.size() / 3;

    if (numVertices == 0)
    {
        return;
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

    mBounds.mRadius = 0.5f * glm::distance(y, z);
    mBounds.mCentre = 0.5f * (y + z);

    for (size_t i = 0; i < numVertices; i++)
    {
        glm::vec3 temp = glm::vec3(mVertices[3 * i], mVertices[3 * i + 1], mVertices[3 * i + 2]);
        float radius = glm::distance(temp, mBounds.mCentre);
        if (radius > mBounds.mRadius)
        {
            mBounds.mRadius = radius;
        }
    }
}