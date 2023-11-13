#include <fstream>

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/Mesh.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"

#include "../../include/graphics/Renderer.h"

#include "tiny_obj_loader.h"

#include <emmintrin.h>
#include <filesystem>
#include <iostream>

using namespace PhysicsEngine;

namespace tinyobj
{
bool operator<(const index_t &l, const index_t &r)
{
    if (l.vertex_index == r.vertex_index)
    {
        if (l.normal_index == r.normal_index)
        {
            if (l.texcoord_index == r.texcoord_index)
            {
                return false;
            }
            else
            {
                return l.texcoord_index < r.texcoord_index;
            }
        }
        else
        {
            return l.normal_index < r.normal_index;
        }
    }
    else
    {
        return l.vertex_index < r.vertex_index;
    }
}
} // namespace tinyobj

Mesh::Mesh(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mName = "Unnamed Asset";

    mSource = "";
    mSourceFilepath = "";
    mDeviceUpdateRequired = false;

    mVertexBuffer = VertexBuffer::create();
    mNormalBuffer = VertexBuffer::create();
    mTexCoordsBuffer = VertexBuffer::create();
    mInstanceModelBuffer = VertexBuffer::create();
    mInstanceColorBuffer = VertexBuffer::create();

    mIndexBuffer = IndexBuffer::create();

    mHandle = MeshHandle::create();

    mHandle->addVertexBuffer(mVertexBuffer, AttribType::Vec3);
    mHandle->addVertexBuffer(mNormalBuffer, AttribType::Vec3);
    mHandle->addVertexBuffer(mTexCoordsBuffer, AttribType::Vec2);
    mHandle->addVertexBuffer(mInstanceModelBuffer, AttribType::Mat4, true);
    mHandle->addVertexBuffer(mInstanceColorBuffer, AttribType::UVec4, true);

    mHandle->addIndexBuffer(mIndexBuffer);
}

Mesh::Mesh(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mName = "Unnamed Asset";

    mSource = "";
    mSourceFilepath = "";
    mDeviceUpdateRequired = false;

    mVertexBuffer = VertexBuffer::create();
    mNormalBuffer = VertexBuffer::create();
    mTexCoordsBuffer = VertexBuffer::create();
    mInstanceModelBuffer = VertexBuffer::create();
    mInstanceColorBuffer = VertexBuffer::create();

    mIndexBuffer = IndexBuffer::create();

    mHandle = MeshHandle::create();

    mHandle->addVertexBuffer(mVertexBuffer, AttribType::Vec3);
    mHandle->addVertexBuffer(mNormalBuffer, AttribType::Vec3);
    mHandle->addVertexBuffer(mTexCoordsBuffer, AttribType::Vec2);
    mHandle->addVertexBuffer(mInstanceModelBuffer, AttribType::Mat4, true);
    mHandle->addVertexBuffer(mInstanceColorBuffer, AttribType::UVec4, true);

    mHandle->addIndexBuffer(mIndexBuffer);
}

Mesh::~Mesh()
{
    delete mVertexBuffer;
    delete mNormalBuffer;
    delete mTexCoordsBuffer;
    delete mInstanceModelBuffer;
    delete mInstanceColorBuffer;

    delete mIndexBuffer;

    delete mHandle;
}

void Mesh::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["name"] = mName;

    out["source"] = mSource;
}

void Mesh::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mName = YAML::getValue<std::string>(in, "name");

    mSource = YAML::getValue<std::string>(in, "source");
    mSourceFilepath = YAML::getValue<std::string>(in, "sourceFilepath"); // dont serialize out
    load(mSourceFilepath);
}

bool Mesh::writeToYAML(const std::string &filepath) const
{
    std::ofstream out;
    out.open(filepath);

    if (!out.is_open())
    {
        return false;
    }

    if (mHide == HideFlag::None)
    {
        YAML::Node n;
        serialize(n);

        YAML::Node assetNode;
        assetNode[getObjectName()] = n;

        out << assetNode;
        out << "\n";
    }
    out.close();

    return true;
}

void Mesh::loadFromYAML(const std::string &filepath)
{
    YAML::Node in = YAML::LoadFile(filepath);

    if (!in.IsMap())
    {
        return;
    }

    for (YAML::const_iterator it = in.begin(); it != in.end(); ++it)
    {
        if (it->first.IsScalar() && it->second.IsMap())
        {
            deserialize(it->second);
        }
    }
}

int Mesh::getType() const
{
    return PhysicsEngine::MESH_TYPE;
}

std::string Mesh::getObjectName() const
{
    return PhysicsEngine::MESH_NAME;
}

Guid Mesh::getGuid() const
{
    return mGuid;
}

Id Mesh::getId() const
{
    return mId;
}

/*void Mesh::load(const std::string &filepath)
{
    if (filepath.empty())
    {
        return;
    }

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filepath, reader_config))
    {
        if (!reader.Error().empty())
        {
            Log::error(reader.Error().c_str());
            return;
        }
    }

    if (!reader.Warning().empty())
    {
        Log::warn(reader.Warning().c_str());
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    //auto &materials = reader.GetMaterials();

    mSubMeshVertexStartIndices.push_back(0);

    size_t vertexCount = 0;
    for (size_t s = 0; s < shapes.size(); s++)
    {
        vertexCount += shapes[s].mesh.indices.size();
    }

    mVertices.resize(3 * vertexCount);
    mNormals.resize(3 * vertexCount);
    mTexCoords.resize(2 * vertexCount);
    mColors.resize(3 * vertexCount);

    size_t vIndex = 0;
    size_t nIndex = 0;
    size_t tIndex = 0;
    size_t cIndex = 0;

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++)
    {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
        {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++)
            {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                mVertices[3 * vIndex + 0] = vx;
                mVertices[3 * vIndex + 1] = vy;
                mVertices[3 * vIndex + 2] = vz;
                vIndex++;

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0)
                {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                    mNormals[3 * nIndex + 0] = nx;
                    mNormals[3 * nIndex + 1] = ny;
                    mNormals[3 * nIndex + 2] = nz;
                    nIndex++;
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0)
                {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];

                    mTexCoords[2 * tIndex + 0] = tx;
                    mTexCoords[2 * tIndex + 1] = ty;
                    tIndex++;
                }
                // Optional: vertex colors
                tinyobj::real_t red = attrib.colors[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t green = attrib.colors[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t blue = attrib.colors[3 * size_t(idx.vertex_index) + 2];

                mColors[3 * cIndex + 0] = red;
                mColors[3 * cIndex + 1] = green;
                mColors[3 * cIndex + 2] = blue;
                cIndex++;

            }
            index_offset += fv;

            // per-face material
            // shapes[s].mesh.material_ids[f];
        }

        mSubMeshVertexStartIndices.push_back((int)mVertices.size());
    }

    //if(vIndex != nIndex)
    //{
        //computeNormals();
        computeNormals_SIMD128();
    //}

    //computeBoundingSphere();
    computeBoundingSphere_SIMD128();

    std::filesystem::path temp = filepath;
    mSource = temp.filename().string();

    mDeviceUpdateRequired = true;
}*/

void Mesh::load(const std::string &filepath)
{
    if (filepath.empty())
    {
        return;
    }

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filepath, reader_config))
    {
        if (!reader.Error().empty())
        {
            Log::error(reader.Error().c_str());
            return;
        }
    }

    if (!reader.Warning().empty())
    {
        Log::warn(reader.Warning().c_str());
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    // auto &materials = reader.GetMaterials();

    mSubMeshStartIndices.resize(shapes.size() + 1, 0);

    // Number unique (vertx_index, normal-index, tecoord_index) triples
    size_t count = 0;
    size_t totalIndices = 0;
    std::map<tinyobj::index_t, size_t> map;
    for (size_t s = 0; s < shapes.size(); s++)
    {
        for (size_t i = 0; i < shapes[s].mesh.indices.size(); i++)
        {
            auto it = map.find(shapes[s].mesh.indices[i]);
            if (it == map.end())
            {
                map.insert(std::pair<tinyobj::index_t, size_t>(shapes[s].mesh.indices[i], count));
                count++;
            }
        }

        totalIndices += shapes[s].mesh.indices.size();

        mSubMeshStartIndices[s + 1] = (int)(totalIndices);
    }

    mIndices.resize(totalIndices);
    mVertices.resize(3 * count);
    mNormals.resize(3 * count);
    mTexCoords.resize(2 * count);

    // Loop over indices and construct vertex, normals, and texcoords arrays
    for (auto const &entry : map)
    {
        tinyobj::index_t key = entry.first;
        size_t index = entry.second;

        tinyobj::real_t vx = attrib.vertices[3 * size_t(key.vertex_index) + 0];
        tinyobj::real_t vy = attrib.vertices[3 * size_t(key.vertex_index) + 1];
        tinyobj::real_t vz = attrib.vertices[3 * size_t(key.vertex_index) + 2];

        mVertices[3 * index + 0] = vx;
        mVertices[3 * index + 1] = vy;
        mVertices[3 * index + 2] = vz;

        // Check if `normal_index` is zero or positive. negative = no normal data
        if (key.normal_index >= 0)
        {
            tinyobj::real_t nx = attrib.normals[3 * size_t(key.normal_index) + 0];
            tinyobj::real_t ny = attrib.normals[3 * size_t(key.normal_index) + 1];
            tinyobj::real_t nz = attrib.normals[3 * size_t(key.normal_index) + 2];

            mNormals[3 * index + 0] = nx;
            mNormals[3 * index + 1] = ny;
            mNormals[3 * index + 2] = nz;
        }

        // Check if `texcoord_index` is zero or positive. negative = no texcoord data
        if (key.texcoord_index >= 0)
        {
            tinyobj::real_t tx = attrib.texcoords[2 * size_t(key.texcoord_index) + 0];
            tinyobj::real_t ty = attrib.texcoords[2 * size_t(key.texcoord_index) + 1];

            mTexCoords[2 * index + 0] = tx;
            mTexCoords[2 * index + 1] = ty;
        }
    }

    // Create indices array
    size_t index = 0;
    for (size_t s = 0; s < shapes.size(); s++)
    {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
        {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++)
            {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                auto it = map.find(idx);
                if (it != map.end())
                {
                    mIndices[index] = static_cast<unsigned int>(it->second);
                    index++;
                }
                else
                {
                    std::cout << "Error" << std::endl;
                }
            }

            index_offset += fv;
        }
    }

    // if(vIndex != nIndex)
    //{
    // computeNormals();
    // computeNormals_SIMD128();
    //}

    // computeBoundingSphere();
    computeBoundingSphere_SIMD128();

    std::filesystem::path temp = filepath;
    mSource = temp.filename().string();

    mVertexCount = mVertices.size() / 3;
    mIndexCount = mIndices.size();

    mDeviceUpdateRequired = true;
}

void Mesh::load(std::vector<float> vertices, std::vector<float> normals, std::vector<float> texCoords,
                std::vector<float> colors, std::vector<unsigned int> indices, std::vector<int> subMeshStartIndices)
{
    mColors = colors;
    load(vertices, normals, texCoords, indices, subMeshStartIndices);
}

void Mesh::load(std::vector<float> vertices, std::vector<float> normals, std::vector<float> texCoords,
                std::vector<unsigned int> indices, std::vector<int> subMeshStartIndices)
{
    mVertices = vertices;
    mNormals = normals;
    mTexCoords = texCoords;
    mIndices = indices;
    mSubMeshStartIndices = subMeshStartIndices;

    if (mVertices.size() != mNormals.size())
    {
        mNormals.resize(mVertices.size());
    }

    if (2 * mVertices.size() != 3 * mTexCoords.size())
    {
        mTexCoords.resize(2 * mVertices.size() / 3);
    }

    // computeBoundingSphere();
    computeBoundingSphere_SIMD128();

    mVertexCount = mVertices.size() / 3;
    mIndexCount = mIndices.size();

    mDeviceUpdateRequired = true;
}

bool Mesh::deviceUpdateRequired() const
{
    return mDeviceUpdateRequired;
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

const std::vector<float> &Mesh::getColors() const
{
    return mColors;
}

const std::vector<unsigned int> &Mesh::getIndices() const
{
    return mIndices;
}

const std::vector<int> &Mesh::getSubMeshStartIndices() const
{
    return mSubMeshStartIndices;
}

size_t Mesh::getVertexCount() const
{
    return mVertexCount;
}

size_t Mesh::getIndexCount() const
{
    return mIndexCount;
}

int Mesh::getSubMeshStartIndex(int subMeshIndex) const
{
    if (subMeshIndex >= mSubMeshStartIndices.size() - 1)
    {
        return -1;
    }

    return mSubMeshStartIndices[subMeshIndex];
}

int Mesh::getSubMeshEndIndex(int subMeshIndex) const
{
    if (subMeshIndex >= mSubMeshStartIndices.size() - 1)
    {
        return -1;
    }

    return mSubMeshStartIndices[subMeshIndex + 1];
}

int Mesh::getSubMeshCount() const
{
    return (int)mSubMeshStartIndices.size() - 1;
}

Sphere Mesh::getBounds() const
{
    return mBounds;
}

MeshHandle *Mesh::getNativeGraphicsHandle() const
{
    return mHandle;
}

VertexBuffer* Mesh::getNativeGraphicsVertexBuffer() const
{
    return mVertexBuffer;
}

VertexBuffer *Mesh::getNativeGraphicsNormallBuffer() const
{
    return mNormalBuffer;
}

VertexBuffer *Mesh::getNativeGraphicsTexCoordsBuffer() const
{
    return mTexCoordsBuffer;
}

VertexBuffer *Mesh::getNativeGraphicsInstanceModelBuffer() const
{
    return mInstanceModelBuffer;
}

VertexBuffer *Mesh::getNativeGraphicsInstanceColorBuffer() const
{
    return mInstanceColorBuffer;
}

IndexBuffer *Mesh::getNativeGraphicsIndexBuffer() const
{
    return mIndexBuffer;
}

void Mesh::setVertices(const std::vector<float> &vertices)
{
    if (vertices.size() / 3 == mVertexCount)
    {
        mVertices = vertices;
        mDeviceUpdateRequired = true;

        // computeBoundingSphere();
        computeBoundingSphere_SIMD128();
    }
}

void Mesh::setNormals(const std::vector<float> &normals)
{
    if (normals.size() / 3 == mVertexCount)
    {
        mNormals = normals;
        mDeviceUpdateRequired = true;
    }
}

void Mesh::setTexCoords(const std::vector<float> &texCoords)
{
    if (3 * (texCoords.size() / 2) == mVertexCount)
    {
        mTexCoords = texCoords;
        mDeviceUpdateRequired = true;
    }
}

void Mesh::setColors(const std::vector<float> &colors)
{
    if (colors.size() / 4 == mVertexCount)
    {
        mColors = colors;
        mDeviceUpdateRequired = true;
    }
}

void Mesh::copyMeshToDevice()
{
    if (mDeviceUpdateRequired)
    {
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

        mInstanceModelBuffer->bind();
        if (mInstanceModelBuffer->getSize() < sizeof(glm::mat4) * Renderer::getRenderer()->INSTANCE_BATCH_SIZE)
        {
            mInstanceModelBuffer->resize(sizeof(glm::mat4) * Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
        }
        // mInstanceModelBuffer->setData(nullptr, 0, sizeof(glm::mat4) * Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
        mInstanceModelBuffer->unbind();

        mInstanceColorBuffer->bind();
        if (mInstanceColorBuffer->getSize() < sizeof(glm::uvec4) * Renderer::getRenderer()->INSTANCE_BATCH_SIZE)
        {
            mInstanceColorBuffer->resize(sizeof(glm::uvec4) * Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
        }
        // mInstanceColorBuffer->setData(nullptr, 0, sizeof(glm::uvec4) * Renderer::getRenderer()->INSTANCE_BATCH_SIZE);
        mInstanceColorBuffer->unbind();

        mIndexBuffer->bind();
        if (mIndexBuffer->getSize() < sizeof(unsigned int) * mIndices.size())
        {
            mIndexBuffer->resize(sizeof(unsigned int) * mIndices.size());
        }
        mIndexBuffer->setData(mIndices.data(), 0, sizeof(unsigned int) * mIndices.size());
        mIndexBuffer->unbind();

        mDeviceUpdateRequired = false;
    }
}

void Mesh::writeMesh()
{
}

void Mesh::computeNormals()
{
    size_t numTriangles = mVertices.size() / 9;

    for (size_t t = 0; t < numTriangles; t++)
    {
        float vx1 = mVertices[9 * t + 0];
        float vy1 = mVertices[9 * t + 1];
        float vz1 = mVertices[9 * t + 2];

        float vx2 = mVertices[9 * t + 3];
        float vy2 = mVertices[9 * t + 4];
        float vz2 = mVertices[9 * t + 5];

        float vx3 = mVertices[9 * t + 6];
        float vy3 = mVertices[9 * t + 7];
        float vz3 = mVertices[9 * t + 8];

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
        // Add the normal 3 times (once for each vertex in triangle)
        mNormals[9 * t + 0] = nx;
        mNormals[9 * t + 1] = ny;
        mNormals[9 * t + 2] = nz;
        mNormals[9 * t + 3] = nx;
        mNormals[9 * t + 4] = ny;
        mNormals[9 * t + 5] = nz;
        mNormals[9 * t + 6] = nx;
        mNormals[9 * t + 7] = ny;
        mNormals[9 * t + 8] = nz;
    }
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

void Mesh::computeNormals_SIMD128()
{
    size_t numTriangles = mVertices.size() / 9;
    size_t numSimdTriangles = numTriangles - (numTriangles % 4);

    std::cout << "numTriangles: " << numTriangles << " numSimdTriangles: " << numSimdTriangles
              << " mName: " << mName << std::endl;

    for (size_t t = 0; t < numSimdTriangles; t += 4)
    {
        __m128 vx1 =
            _mm_set_ps(mVertices[9 * t + 0], mVertices[9 * t + 9], mVertices[9 * t + 18], mVertices[9 * t + 27]);
        __m128 vy1 =
            _mm_set_ps(mVertices[9 * t + 1], mVertices[9 * t + 10], mVertices[9 * t + 19], mVertices[9 * t + 28]);
        __m128 vz1 =
            _mm_set_ps(mVertices[9 * t + 2], mVertices[9 * t + 11], mVertices[9 * t + 20], mVertices[9 * t + 29]);

        __m128 vx2 =
            _mm_set_ps(mVertices[9 * t + 3], mVertices[9 * t + 12], mVertices[9 * t + 21], mVertices[9 * t + 30]);
        __m128 vy2 =
            _mm_set_ps(mVertices[9 * t + 4], mVertices[9 * t + 13], mVertices[9 * t + 22], mVertices[9 * t + 31]);
        __m128 vz2 =
            _mm_set_ps(mVertices[9 * t + 5], mVertices[9 * t + 14], mVertices[9 * t + 23], mVertices[9 * t + 32]);

        __m128 vx3 =
            _mm_set_ps(mVertices[9 * t + 6], mVertices[9 * t + 15], mVertices[9 * t + 24], mVertices[9 * t + 33]);
        __m128 vy3 =
            _mm_set_ps(mVertices[9 * t + 7], mVertices[9 * t + 16], mVertices[9 * t + 25], mVertices[9 * t + 34]);
        __m128 vz3 =
            _mm_set_ps(mVertices[9 * t + 8], mVertices[9 * t + 17], mVertices[9 * t + 26], mVertices[9 * t + 35]);

        // Calculate p vector
        __m128 px = _mm_sub_ps(vx2, vx1);
        __m128 py = _mm_sub_ps(vy2, vy1);
        __m128 pz = _mm_sub_ps(vz2, vz1);
        // Calculate q vector
        __m128 qx = _mm_sub_ps(vx3, vx1);
        __m128 qy = _mm_sub_ps(vy3, vy1);
        __m128 qz = _mm_sub_ps(vz3, vz1);

        // Calculate normal (p x q)
        // i  j  k
        // px py pz
        // qx qy qz
        __m128 nx = _mm_sub_ps(_mm_mul_ps(py, qz), _mm_mul_ps(pz, qy));
        __m128 ny = _mm_sub_ps(_mm_mul_ps(pz, qx), _mm_mul_ps(px, qz));
        __m128 nz = _mm_sub_ps(_mm_mul_ps(px, qy), _mm_mul_ps(py, qx));

        // Scale to unit vector
        __m128 s = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(nx, nx), _mm_add_ps(_mm_mul_ps(ny, ny), _mm_mul_ps(nz, nz))));
        nx = _mm_div_ps(nx, s);
        ny = _mm_div_ps(ny, s);
        nz = _mm_div_ps(nz, s);

        // Add the normal 3 times (once for each vertex in triangle)
        alignas(16) float nx_t[4];
        alignas(16) float ny_t[4];
        alignas(16) float nz_t[4];

        _mm_store_ps(nx_t, nx);
        _mm_store_ps(ny_t, ny);
        _mm_store_ps(nz_t, nz);

        mNormals[9 * t + 0] = nx_t[3];
        mNormals[9 * t + 1] = ny_t[3];
        mNormals[9 * t + 2] = nz_t[3];
        mNormals[9 * t + 3] = nx_t[3];
        mNormals[9 * t + 4] = ny_t[3];
        mNormals[9 * t + 5] = nz_t[3];
        mNormals[9 * t + 6] = nx_t[3];
        mNormals[9 * t + 7] = ny_t[3];
        mNormals[9 * t + 8] = nz_t[3];

        mNormals[9 * t + 9] = nx_t[2];
        mNormals[9 * t + 10] = ny_t[2];
        mNormals[9 * t + 11] = nz_t[2];
        mNormals[9 * t + 12] = nx_t[2];
        mNormals[9 * t + 13] = ny_t[2];
        mNormals[9 * t + 14] = nz_t[2];
        mNormals[9 * t + 15] = nx_t[2];
        mNormals[9 * t + 16] = ny_t[2];
        mNormals[9 * t + 17] = nz_t[2];

        mNormals[9 * t + 18] = nx_t[1];
        mNormals[9 * t + 19] = ny_t[1];
        mNormals[9 * t + 20] = nz_t[1];
        mNormals[9 * t + 21] = nx_t[1];
        mNormals[9 * t + 22] = ny_t[1];
        mNormals[9 * t + 23] = nz_t[1];
        mNormals[9 * t + 24] = nx_t[1];
        mNormals[9 * t + 25] = ny_t[1];
        mNormals[9 * t + 26] = nz_t[1];

        mNormals[9 * t + 27] = nx_t[0];
        mNormals[9 * t + 28] = ny_t[0];
        mNormals[9 * t + 29] = nz_t[0];
        mNormals[9 * t + 30] = nx_t[0];
        mNormals[9 * t + 31] = ny_t[0];
        mNormals[9 * t + 32] = nz_t[0];
        mNormals[9 * t + 33] = nx_t[0];
        mNormals[9 * t + 34] = ny_t[0];
        mNormals[9 * t + 35] = nz_t[0];
    }

    for (size_t t = numSimdTriangles; t < numTriangles; t++)
    {
        float vx1 = mVertices[9 * t + 0];
        float vy1 = mVertices[9 * t + 1];
        float vz1 = mVertices[9 * t + 2];

        float vx2 = mVertices[9 * t + 3];
        float vy2 = mVertices[9 * t + 4];
        float vz2 = mVertices[9 * t + 5];

        float vx3 = mVertices[9 * t + 6];
        float vy3 = mVertices[9 * t + 7];
        float vz3 = mVertices[9 * t + 8];

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

        // Add the normal 3 times (once for each vertex in triangle)
        mNormals[9 * t + 0] = nx;
        mNormals[9 * t + 1] = ny;
        mNormals[9 * t + 2] = nz;
        mNormals[9 * t + 3] = nx;
        mNormals[9 * t + 4] = ny;
        mNormals[9 * t + 5] = nz;
        mNormals[9 * t + 6] = nx;
        mNormals[9 * t + 7] = ny;
        mNormals[9 * t + 8] = nz;
    }
}

void Mesh::computeBoundingSphere_SIMD128()
{
    mBounds.mRadius = 0.0f;
    mBounds.mCentre = glm::vec3(0.0f, 0.0f, 0.0f);

    size_t numVertices = mVertices.size() / 3;

    if (numVertices == 0)
    {
        return;
    }

    size_t numSimdVertices = numVertices - (numVertices % 4);

    // Ritter algorithm for bounding sphere
    // find furthest point from first vertex
    // float x_x = mVertices[0];
    // float x_y = mVertices[1];
    // float x_z = mVertices[2];
    __m128 x_x = _mm_set_ps(mVertices[0], mVertices[0], mVertices[0], mVertices[0]);
    __m128 x_y = _mm_set_ps(mVertices[1], mVertices[1], mVertices[1], mVertices[1]);
    __m128 x_z = _mm_set_ps(mVertices[2], mVertices[2], mVertices[2], mVertices[2]);

    // float y_x = x_x;
    // float y_y = x_y;
    // float y_z = x_z;
    // float maxDistance = 0.0f;
    __m128 y_x = x_x;
    __m128 y_y = x_y;
    __m128 y_z = x_z;
    __m128 maxDistance = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
    for (size_t i = 0; i < numSimdVertices; i += 4)
    {
        // float temp_x = mVertices[3 * i];
        // float temp_y = mVertices[3 * i + 1];
        // float temp_z = mVertices[3 * i + 2];
        __m128 temp_x =
            _mm_set_ps(mVertices[3 * i + 0], mVertices[3 * i + 3], mVertices[3 * i + 6], mVertices[3 * i + 9]);
        __m128 temp_y =
            _mm_set_ps(mVertices[3 * i + 1], mVertices[3 * i + 4], mVertices[3 * i + 7], mVertices[3 * i + 10]);
        __m128 temp_z =
            _mm_set_ps(mVertices[3 * i + 2], mVertices[3 * i + 5], mVertices[3 * i + 8], mVertices[3 * i + 11]);

        // calculate distance between x and temp
        __m128 xmt_x = _mm_sub_ps(x_x, temp_x);
        __m128 xmt_y = _mm_sub_ps(x_y, temp_y);
        __m128 xmt_z = _mm_sub_ps(x_z, temp_z);

        xmt_x = _mm_mul_ps(xmt_x, xmt_x);
        xmt_y = _mm_mul_ps(xmt_y, xmt_y);
        xmt_z = _mm_mul_ps(xmt_z, xmt_z);

        // float distance = sqrt((x_x - temp_x) * (x_x - temp_x) +
        //                      (x_y - temp_y) * (x_y - temp_y) +
        //                      (x_z - temp_z) * (x_z - temp_z));
        __m128 distance = _mm_sqrt_ps(_mm_add_ps(xmt_x, _mm_add_ps(xmt_y, xmt_z)));

        // if (distance > maxDistance)
        //{
        //    y_x = temp_x;
        //    y_y = temp_y;
        //    y_z = temp_z;
        //    maxDistance = distance;
        //}
        // if (x) y=temp; else y=y; ==> y=y+x*(temp-y);
        __m128 condition = _mm_and_ps(_mm_set1_ps(1), _mm_cmpgt_ps(distance, maxDistance));
        y_x = _mm_add_ps(y_x, _mm_mul_ps(condition, _mm_sub_ps(temp_x, y_x)));
        y_y = _mm_add_ps(y_y, _mm_mul_ps(condition, _mm_sub_ps(temp_y, y_y)));
        y_z = _mm_add_ps(y_z, _mm_mul_ps(condition, _mm_sub_ps(temp_z, y_z)));
        maxDistance = _mm_add_ps(maxDistance, _mm_mul_ps(condition, _mm_sub_ps(distance, maxDistance)));
    }

    alignas(16) float y2_x[4];
    alignas(16) float y2_y[4];
    alignas(16) float y2_z[4];

    _mm_store_ps(y2_x, y_x);
    _mm_store_ps(y2_y, y_y);
    _mm_store_ps(y2_z, y_z);

    glm::vec3 x = glm::vec3(mVertices[0], mVertices[1], mVertices[2]);
    glm::vec3 y = x;
    float maxDistance2 = 0.0f;
    for (size_t i = 0; i < 4; i++)
    {
        glm::vec3 temp = glm::vec3(y2_x[i], y2_y[i], y2_z[i]);
        float distance = glm::distance(x, temp);
        if (distance > maxDistance2)
        {
            y = temp;
            maxDistance2 = distance;
        }
    }

    for (size_t i = numSimdVertices; i < numVertices; i++)
    {
        glm::vec3 temp = glm::vec3(mVertices[3 * i], mVertices[3 * i + 1], mVertices[3 * i + 2]);
        float distance = glm::distance(x, temp);
        if (distance > maxDistance2)
        {
            y = temp;
            maxDistance2 = distance;
        }
    }

    y_x = _mm_set_ps(y.x, y.x, y.x, y.x);
    y_y = _mm_set_ps(y.y, y.y, y.y, y.y);
    y_z = _mm_set_ps(y.z, y.z, y.z, y.z);

    // now find furthest point from y
    // float z_x = y_x;
    // float z_y = y_y;
    // float z_z = y_z;
    // maxDistance = 0.0f;
    __m128 z_x = y_x;
    __m128 z_y = y_y;
    __m128 z_z = y_z;
    maxDistance = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);
    for (size_t i = 0; i < numSimdVertices; i += 4)
    {
        // float temp_x = mVertices[3 * i];
        // float temp_y = mVertices[3 * i + 1];
        // float temp_z = mVertices[3 * i + 2];
        __m128 temp_x =
            _mm_set_ps(mVertices[3 * i + 0], mVertices[3 * i + 3], mVertices[3 * i + 6], mVertices[3 * i + 9]);
        __m128 temp_y =
            _mm_set_ps(mVertices[3 * i + 1], mVertices[3 * i + 4], mVertices[3 * i + 7], mVertices[3 * i + 10]);
        __m128 temp_z =
            _mm_set_ps(mVertices[3 * i + 2], mVertices[3 * i + 5], mVertices[3 * i + 8], mVertices[3 * i + 11]);

        // calculate distance between y and temp
        __m128 ymt_x = _mm_sub_ps(y_x, temp_x);
        __m128 ymt_y = _mm_sub_ps(y_y, temp_y);
        __m128 ymt_z = _mm_sub_ps(y_z, temp_z);

        ymt_x = _mm_mul_ps(ymt_x, ymt_x);
        ymt_y = _mm_mul_ps(ymt_y, ymt_y);
        ymt_z = _mm_mul_ps(ymt_z, ymt_z);

        // float distance = sqrt((y_x - temp_x) * (y_x - temp_x) +
        //                      (y_y - temp_y) * (y_y - temp_y) +
        //                      (y_z - temp_z) * (y_z - temp_z));
        __m128 distance = _mm_sqrt_ps(_mm_add_ps(ymt_x, _mm_add_ps(ymt_y, ymt_z)));

        // if (distance > maxDistance)
        // {
        //     z_x = temp_x;
        //     z_y = temp_y;
        //     z_z = temp_z;
        //     maxDistance = distance;
        // }
        // if (x) z=temp; else z=z; ==> z=z+x*(temp-z);
        __m128 condition = _mm_and_ps(_mm_set1_ps(1), _mm_cmpgt_ps(distance, maxDistance));
        z_x = _mm_add_ps(z_x, _mm_mul_ps(condition, _mm_sub_ps(temp_x, z_x)));
        z_y = _mm_add_ps(z_y, _mm_mul_ps(condition, _mm_sub_ps(temp_y, z_y)));
        z_z = _mm_add_ps(z_z, _mm_mul_ps(condition, _mm_sub_ps(temp_z, z_z)));
        maxDistance = _mm_add_ps(maxDistance, _mm_mul_ps(condition, _mm_sub_ps(distance, maxDistance)));
    }

    alignas(16) float z2_x[4];
    alignas(16) float z2_y[4];
    alignas(16) float z2_z[4];

    _mm_store_ps(z2_x, z_x);
    _mm_store_ps(z2_y, z_y);
    _mm_store_ps(z2_z, z_z);

    glm::vec3 z = y;
    maxDistance2 = 0.0f;
    for (size_t i = 0; i < 4; i++)
    {
        glm::vec3 temp = glm::vec3(z2_x[i], z2_y[i], z2_z[i]);
        float distance = glm::distance(y, temp);
        if (distance > maxDistance2)
        {
            z = temp;
            maxDistance2 = distance;
        }
    }

    for (size_t i = numSimdVertices; i < numVertices; i++)
    {
        glm::vec3 temp = glm::vec3(mVertices[3 * i], mVertices[3 * i + 1], mVertices[3 * i + 2]);
        float distance = glm::distance(y, temp);
        if (distance > maxDistance2)
        {
            z = temp;
            maxDistance2 = distance;
        }
    }

    mBounds.mRadius = 0.5f * glm::distance(y, z);
    mBounds.mCentre = 0.5f * (y + z);

    __m128 radius = _mm_set_ps(mBounds.mRadius, mBounds.mRadius, mBounds.mRadius, mBounds.mRadius);
    __m128 centre_x = _mm_set_ps(mBounds.mCentre.x, mBounds.mCentre.x, mBounds.mCentre.x, mBounds.mCentre.x);
    __m128 centre_y = _mm_set_ps(mBounds.mCentre.y, mBounds.mCentre.y, mBounds.mCentre.y, mBounds.mCentre.y);
    __m128 centre_z = _mm_set_ps(mBounds.mCentre.z, mBounds.mCentre.z, mBounds.mCentre.z, mBounds.mCentre.z);

    for (size_t i = 0; i < numSimdVertices; i += 4)
    {
        // float temp_x = mVertices[3 * i];
        // float temp_y = mVertices[3 * i + 1];
        // float temp_z = mVertices[3 * i + 2];
        __m128 temp_x =
            _mm_set_ps(mVertices[3 * i + 0], mVertices[3 * i + 3], mVertices[3 * i + 6], mVertices[3 * i + 9]);
        __m128 temp_y =
            _mm_set_ps(mVertices[3 * i + 1], mVertices[3 * i + 4], mVertices[3 * i + 7], mVertices[3 * i + 10]);
        __m128 temp_z =
            _mm_set_ps(mVertices[3 * i + 2], mVertices[3 * i + 5], mVertices[3 * i + 8], mVertices[3 * i + 11]);

        // calculate distance between centre and temp
        __m128 centremt_x = _mm_sub_ps(centre_x, temp_x);
        __m128 centremt_y = _mm_sub_ps(centre_y, temp_y);
        __m128 centremt_z = _mm_sub_ps(centre_z, temp_z);

        centremt_x = _mm_mul_ps(centremt_x, centremt_x);
        centremt_y = _mm_mul_ps(centremt_y, centremt_y);
        centremt_z = _mm_mul_ps(centremt_z, centremt_z);

        // float distance = sqrt((centre_x - temp_x) * (centre_x - temp_x) +
        //                       (centre_y - temp_y) * (centre_y - temp_y) +
        //                       (centre_z - temp_z) * (centre_z - temp_z));
        __m128 distance = _mm_sqrt_ps(_mm_add_ps(centremt_x, _mm_add_ps(centremt_y, centremt_z)));

        // if (distance > mBounds.mRadius)
        // {
        //     mBounds.mRadius = distance;
        // }
        radius = _mm_max_ps(distance, radius);
    }

    alignas(16) float radius2[4];
    _mm_store_ps(radius2, radius);

    for (size_t i = 0; i < 4; i++)
    {
        if (radius2[i] > mBounds.mRadius)
        {
            mBounds.mRadius = radius2[i];
        }
    }

    for (size_t i = numSimdVertices; i < numVertices; i++)
    {
        glm::vec3 temp = glm::vec3(mVertices[3 * i], mVertices[3 * i + 1], mVertices[3 * i + 2]);
        float distance = glm::distance(temp, mBounds.mCentre);
        if (distance > mBounds.mRadius)
        {
            mBounds.mRadius = distance;
        }
    }
}