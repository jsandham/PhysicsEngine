#include "../../include/core/Mesh.h"
#include "../../include/core/Log.h"
#include "../../include/core/Serialization.h"
#include "../../include/graphics/Graphics.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace PhysicsEngine;

Mesh::Mesh(World* world) : Asset(world)
{
    mSource = "";
    mCreated = false;
    mChanged = false;
}

Mesh::Mesh(World* world, Guid id) : Asset(world, id)
{
    mSource = "";
    mCreated = false;
    mChanged = false;
}

Mesh::~Mesh()
{
}

void Mesh::serialize(YAML::Node &out) const
{
    Asset::serialize(out);

    out["source"] = mSource;
}

void Mesh::deserialize(const YAML::Node &in)
{
    Asset::deserialize(in);

    mSource = YAML::getValue<std::string>(in, "source");
    load(mSource);
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
    if (filepath.empty())
    {
        return;
    }

    /*obj_mesh mesh;

    if (obj_load(filepath, mesh))
    {
        mVertices = mesh.mVertices;
        mNormals = mesh.mNormals;
        mTexCoords = mesh.mTexCoords;
        mSubMeshVertexStartIndices = mesh.mSubMeshVertexStartIndices;

        if (mVertices.size() != mNormals.size())
        {
            mNormals.resize(mVertices.size());
        }

        if (2 * mVertices.size() != 3 * mTexCoords.size())
        {
            mTexCoords.resize(2 * mVertices.size() / 3);
        }

        computeBoundingSphere();

        mCreated = false;
    }
    else
    {
        Log::error(("Could not load obj mesh " + filepath + "\n").c_str());
    }*/

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filepath, reader_config)) {
        if (!reader.Error().empty()) {
            Log::error(reader.Error().c_str());
            return;
        }
    }

    if (!reader.Warning().empty()) {
        Log::warn(reader.Warning().c_str());
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    mSubMeshVertexStartIndices.push_back(0);

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                mVertices.push_back(vx);
                mVertices.push_back(vy);
                mVertices.push_back(vz);

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                    mNormals.push_back(nx);
                    mNormals.push_back(ny);
                    mNormals.push_back(nz);
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];

                    mTexCoords.push_back(tx);
                    mTexCoords.push_back(ty);
                }
                // Optional: vertex colors
                tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            
                mColors.push_back(red);
                mColors.push_back(green);
                mColors.push_back(blue);
            }
            index_offset += fv;

            // per-face material
            //shapes[s].mesh.material_ids[f];
        }

        mSubMeshVertexStartIndices.push_back((int)mVertices.size());
    }

    if (mNormals.size() != mVertices.size()) {
        // Ensure there are no normals loaded
        mNormals.clear();
        float nx = 0.f, ny = 0.f, nz = 0.0f; // normal for current triangle
        float vx1 = 0.f, vx2 = 0.f, vx3 = 0.f; // vertex 1
        float vy1 = 0.f, vy2 = 0.f, vy3 = 0.f; // vertex 2
        float vz1 = 0.f, vz2 = 0.f, vz3 = 0.f; // vertex 3
        for (size_t v = 0; v < mVertices.size() / 3; v++) {
            float x = mVertices[3 * v];
            float y = mVertices[3 * v + 1];
            float z = mVertices[3 * v + 2];
            switch (v % 3) {
            case 0:
                // Defining first point in triangle
                vx1 = x;
                vy1 = y;
                vz1 = z;
                break;
            case 1:
                // Defining second point in triangle
                vx2 = x;
                vy2 = y;
                vz2 = z;
                break;
            case 2:
                // Defining third point in a triangle
                vx3 = x;
                vy3 = y;
                vz3 = z;

                float qx = 0.0f, qy = 0.0f, qz = 0.0f;
                float px = 0.0f, py = 0.0f, pz = 0.0f;
                // Calculate q vector
                qx = vx2 - vx1;
                qy = vy2 - vy1;
                qz = vz2 - vz1;
                // Calculate p vector
                px = vx3 - vx1;
                py = vy3 - vy1;
                pz = vz3 - vz1;
                // Calculate normal
                nx = py * qz - pz * qy;
                ny = pz * qx - px * qz;
                nz = px * qy - py * qx;
                // Scale to unit vector
                float s = sqrt(nx * nx + ny * ny + nz * nz);
                nx /= s;
                ny /= s;
                nz /= s;
                // Add the normal 3 times (once for each vertex)
                for (int j = 0; j < 3; j++) {
                    mNormals.push_back(nx);
                    mNormals.push_back(ny);
                    mNormals.push_back(nz);
                }
                break;
            }
        }
    }

    computeBoundingSphere();

    mCreated = false;

    mSource = filepath;
}

void Mesh::load(std::vector<float> vertices, std::vector<float> normals, std::vector<float> texCoords,
    std::vector<float> colors, std::vector<int> subMeshStartIndices)
{
    mColors = colors;
    load(vertices, normals, texCoords, subMeshStartIndices);
}

void Mesh::load(std::vector<float> vertices, std::vector<float> normals, std::vector<float> texCoords,
                std::vector<int> subMeshStartIndices)
{
    mVertices = vertices;
    mNormals = normals;
    mTexCoords = texCoords;
    mSubMeshVertexStartIndices = subMeshStartIndices;

    if (mVertices.size() != mNormals.size())
    {
        mNormals.resize(mVertices.size());
    }

    if (2 * mVertices.size() != 3 * mTexCoords.size())
    {
        mTexCoords.resize(2 * mVertices.size() / 3);
    }

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

const std::vector<float>& Mesh::getColors() const
{
    return mColors;
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

void Mesh::setColors(const std::vector<float>& colors)
{
    mColors = colors;

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