#ifndef MESH_H__
#define MESH_H__

#include <vector>

#include "Asset.h"
#include "Sphere.h"
#include "../graphics/VertexBuffer.h"
#include "../graphics/IndexBuffer.h"
#include "../graphics/MeshHandle.h"

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

namespace PhysicsEngine
{
class Mesh : public Asset
{
  private:
    std::string mSource;
    std::string mSourceFilepath;

    std::vector<float> mVertices;
    std::vector<float> mNormals;
    std::vector<float> mTexCoords;
    std::vector<float> mColors;
    std::vector<unsigned int> mIndices;

    std::vector<int> mSubMeshVertexStartIndices;
    std::vector<int> mSubMeshStartIndices;
    
    Sphere mBounds;

    MeshHandle *mHandle;
    VertexBuffer *mVertexBuffer;
    VertexBuffer *mNormalBuffer;
    VertexBuffer *mTexCoordsBuffer;
    VertexBuffer *mInstanceModelBuffer;
    VertexBuffer *mInstanceColorBuffer;
    IndexBuffer *mIndexBuffer;
    bool mDeviceUpdateRequired;

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
              std::vector<float> colors, std::vector<unsigned int> indices, std::vector<int> subMeshStartIndices);
    void load(std::vector<float> vertices, std::vector<float> normals, std::vector<float> texCoords,
              std::vector<unsigned int> indices, std::vector<int> subMeshStartIndices);

    bool deviceUpdateRequired() const;

    const std::vector<float> &getVertices() const;
    const std::vector<float> &getNormals() const;
    const std::vector<float> &getTexCoords() const;
    const std::vector<float> &getColors() const;
    const std::vector<unsigned int> &getIndices() const;
    const std::vector<int> &getSubMeshStartIndices() const;
    int getSubMeshStartIndex(int subMeshIndex) const;
    int getSubMeshEndIndex(int subMeshIndex) const;
    int getSubMeshCount() const;
    Sphere getBounds() const;
    MeshHandle* getNativeGraphicsHandle() const;
    VertexBuffer* getNativeGraphicsInstanceModelBuffer() const;
    VertexBuffer* getNativeGraphicsInstanceColorBuffer() const;

    void setVertices(const std::vector<float> &vertices);
    void setNormals(const std::vector<float> &normals);
    void setTexCoords(const std::vector<float> &texCoords);
    void setColors(const std::vector<float> &colors);

    void copyMeshToDevice();
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