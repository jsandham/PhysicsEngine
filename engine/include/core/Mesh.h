#ifndef MESH_H__
#define MESH_H__

#include <vector>

#include "glm.h"
#include "SerializationEnums.h"
#include "Sphere.h"
#include "AssetEnums.h"
#include "Guid.h"
#include "Id.h"

#include "../graphics/IndexBuffer.h"
#include "../graphics/MeshHandle.h"
#include "../graphics/VertexBuffer.h"

namespace PhysicsEngine
{
class World;

class Mesh
{
  private:
    Guid mGuid;
    Id mId;
    World *mWorld;

    std::string mSource;
    std::string mSourceFilepath;

    std::vector<float> mVertices;
    std::vector<float> mNormals;
    std::vector<float> mTexCoords;
    std::vector<float> mColors;
    std::vector<unsigned int> mIndices;

    size_t mVertexCount;
    size_t mIndexCount;

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

    friend class World;

  public:
    std::string mName;
    HideFlag mHide;

  public:
    Mesh(World *world, const Id &id);
    Mesh(World *world, const Guid &guid, const Id &id);
    ~Mesh();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    bool writeToYAML(const std::string &filepath) const;
    void loadFromYAML(const std::string &filepath);

    int getType() const;
    std::string getObjectName() const;

    Guid getGuid() const;
    Id getId() const;

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

    size_t getVertexCount() const;
    size_t getNormalCount() const;
    size_t getTexCoordCount() const;
    size_t getIndexCount() const;

    int getSubMeshStartIndex(int subMeshIndex) const;
    int getSubMeshEndIndex(int subMeshIndex) const;
    int getSubMeshCount() const;
    Sphere getBounds() const;
    MeshHandle *getNativeGraphicsHandle() const;
    VertexBuffer *getNativeGraphicsVertexBuffer() const;
    VertexBuffer *getNativeGraphicsNormallBuffer() const;
    VertexBuffer *getNativeGraphicsTexCoordsBuffer() const;
    VertexBuffer *getNativeGraphicsInstanceModelBuffer() const;
    VertexBuffer *getNativeGraphicsInstanceColorBuffer() const;
    IndexBuffer *getNativeGraphicsIndexBuffer() const;

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

} // namespace PhysicsEngine

#endif