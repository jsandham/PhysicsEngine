#include "../../include/core/Mesh.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Graphics.h"

#include "tiny_obj_loader.h"

#include <filesystem>
#include <emmintrin.h>
#include <iostream>

using namespace PhysicsEngine;

Mesh::Mesh(World *world) : Asset(world)
{
    mSource = "";
    mSourceFilepath = "";
    mCreated = false;
    mChanged = false;
}

Mesh::Mesh(World *world, const Guid& id) : Asset(world, id)
{
    mSource = "";
    mSourceFilepath = "";
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
    mSourceFilepath = YAML::getValue<std::string>(in, "sourceFilepath"); // dont serialize out
    load(mSourceFilepath);
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

    if(vIndex != nIndex)
    {
        nIndex = 0;

        float nx = 0.f, ny = 0.f, nz = 0.0f;   // normal for current triangle
        float vx1 = 0.f, vx2 = 0.f, vx3 = 0.f; // vertex 1
        float vy1 = 0.f, vy2 = 0.f, vy3 = 0.f; // vertex 2
        float vz1 = 0.f, vz2 = 0.f, vz3 = 0.f; // vertex 3
        for (size_t v = 0; v < mVertices.size() / 3; v++)
        {
            float x = mVertices[3 * v];
            float y = mVertices[3 * v + 1];
            float z = mVertices[3 * v + 2];
            switch (v % 3)
            {
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

                float px = 0.0f, py = 0.0f, pz = 0.0f;
                float qx = 0.0f, qy = 0.0f, qz = 0.0f;
                // Calculate p vector
                px = vx2 - vx1;
                py = vy2 - vy1;
                pz = vz2 - vz1;
                // Calculate q vector
                qx = vx3 - vx1;
                qy = vy3 - vy1;
                qz = vz3 - vz1;

                // Calculate normal (p x q)
                // i  j  k 
                // px py pz
                // qx qy qz
                nx = py * qz - pz * qy;
                ny = pz * qx - px * qz;
                nz = px * qy - py * qx;
                // Scale to unit vector
                float s = sqrt(nx * nx + ny * ny + nz * nz);
                nx /= s;
                ny /= s;
                nz /= s;
                // Add the normal 3 times (once for each vertex)
                for (int j = 0; j < 3; j++)
                {
                    mNormals[3 * nIndex + 0] = nx;
                    mNormals[3 * nIndex + 1] = ny;
                    mNormals[3 * nIndex + 2] = nz;

                    nIndex++;
                }
                break;
            }
        }
    }

    //computeBoundingSphere();
    computeBoundingSphere_SIMD128();

    std::filesystem::path temp = filepath;
    mSource = temp.filename().string();

    mCreated = false;
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

    //computeBoundingSphere();
    computeBoundingSphere_SIMD128();

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

const std::vector<float> &Mesh::getColors() const
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

unsigned int Mesh::getNativeGraphicsVAO() const
{
    return mVao;
}

unsigned int Mesh::getNativeGraphicsVBO(MeshVBO meshVBO) const
{
    return mVbo[static_cast<int>(meshVBO)];
}

void Mesh::setVertices(const std::vector<float> &vertices)
{
    mVertices = vertices;
    //computeBoundingSphere();
    computeBoundingSphere_SIMD128();

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

void Mesh::setColors(const std::vector<float> &colors)
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

    Graphics::createMesh(mVertices, mNormals, mTexCoords, &mVao, &mVbo[0], &mVbo[1], &mVbo[2], &mVbo[3], &mVbo[4]);

    mCreated = true;
}

void Mesh::destroy()
{
    if (!mCreated)
    {
        return;
    }

    Graphics::destroyMesh(&mVao, &mVbo[0], &mVbo[1], &mVbo[2], &mVbo[3], &mVbo[4]);

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

void Mesh::computeBoundingSphere_SIMD128()
{
    mBounds.mRadius = 0.0f;
    mBounds.mCentre = glm::vec3(0.0f, 0.0f, 0.0f);

    size_t numVertices = mVertices.size() / 3;

    if (numVertices == 0){ return; }

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
        __m128 temp_x = _mm_set_ps(mVertices[3 * i + 0], mVertices[3 * i + 3], mVertices[3 * i + 6], mVertices[3 * i + 9]);
        __m128 temp_y = _mm_set_ps(mVertices[3 * i + 1], mVertices[3 * i + 4], mVertices[3 * i + 7], mVertices[3 * i + 10]);
        __m128 temp_z = _mm_set_ps(mVertices[3 * i + 2], mVertices[3 * i + 5], mVertices[3 * i + 8], mVertices[3 * i + 11]);

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

    float y2_x[4];
    float y2_y[4];
    float y2_z[4];

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
        __m128 temp_x = _mm_set_ps(mVertices[3 * i + 0], mVertices[3 * i + 3], mVertices[3 * i + 6], mVertices[3 * i + 9]);
        __m128 temp_y = _mm_set_ps(mVertices[3 * i + 1], mVertices[3 * i + 4], mVertices[3 * i + 7], mVertices[3 * i + 10]);
        __m128 temp_z = _mm_set_ps(mVertices[3 * i + 2], mVertices[3 * i + 5], mVertices[3 * i + 8], mVertices[3 * i + 11]);

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

    float z2_x[4];
    float z2_y[4];
    float z2_z[4];

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

    float radius2[4];
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