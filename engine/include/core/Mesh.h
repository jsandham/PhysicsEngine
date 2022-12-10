#ifndef MESH_H__
#define MESH_H__

#include <vector>

#include "Asset.h"
#include "Sphere.h"
#include "../graphics/VertexBuffer.h"

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

namespace PhysicsEngine
{
enum class MeshVBO
{
    Vertices,
    Normals,
    TexCoords,
    InstanceModel,
    InstanceColor
};

//class VertexArray
//{
//protected:
//    std::vector<VertexBuffer*> mBuffers;
//public:
//    VertexArray();
//    virtual ~VertexArray() = 0;
//
//    virtual void* get() = 0;
//
//    VertexBuffer* getBuffer(size_t index)
//    {
//        return mBuffers[index];
//    }
//
//    void push(VertexBuffer* buffer)
//    {
//        mBuffers.push(buffer);
//    }
//
//    static VertexArray* create();
//};
//
//
//class OpenGLVertexArray : public VertexArray
//{
//public:
//    unsigned int mVao;
//
//    void* get() override;
//};

class Mesh : public Asset
{
  private:
    std::string mSource;
    std::string mSourceFilepath;
    std::vector<float> mVertices;
    std::vector<float> mNormals;
    std::vector<float> mTexCoords;
    std::vector<float> mColors;
    std::vector<int> mSubMeshVertexStartIndices;
    unsigned int mVao;
    VertexBuffer* mVbo[5];
    Sphere mBounds;
    bool mCreated;
    bool mChanged;

  public:
    Mesh(World *world, const Id &id);
    Mesh(World *world, const Guid &guid, const Id &id);
    ~Mesh();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void load(const std::string &filename);
    void load(std::vector<float> vertices, std::vector<float> normals, std::vector<float> texCoords,
              std::vector<float> colors, std::vector<int> subMeshStartIndices);
    void load(std::vector<float> vertices, std::vector<float> normals, std::vector<float> texCoords,
              std::vector<int> subMeshStartIndices);

    bool isCreated() const;
    bool isChanged() const;

    const std::vector<float> &getVertices() const;
    const std::vector<float> &getNormals() const;
    const std::vector<float> &getTexCoords() const;
    const std::vector<float> &getColors() const;
    const std::vector<int> &getSubMeshStartIndices() const;
    int getSubMeshStartIndex(int subMeshIndex) const;
    int getSubMeshEndIndex(int subMeshIndex) const;
    int getSubMeshCount() const;
    Sphere getBounds() const;
    unsigned int getNativeGraphicsVAO() const;
    void* getNativeGraphicsVBO(MeshVBO meshVBO) const;

    void setVertices(const std::vector<float> &vertices);
    void setNormals(const std::vector<float> &normals);
    void setTexCoords(const std::vector<float> &texCoords);
    void setColors(const std::vector<float> &colors);

    void create();
    void destroy();
    void writeMesh();

  private:
    void computeNormals();
    void computeBoundingSphere();
    void computeNormals_SIMD128();
    void computeBoundingSphere_SIMD128();
};

template <> struct AssetType<Mesh>
{
    static constexpr int type = PhysicsEngine::MESH_TYPE;
};

template <> struct IsAssetInternal<Mesh>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif