#ifndef MESH_H__
#define MESH_H__

#include <string>
#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "Asset.h"
#include "Guid.h"
#include "Sphere.h"

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
class Mesh : public Asset
{
  private:
    std::string mSource;
    std::vector<float> mVertices;
    std::vector<float> mNormals;
    std::vector<float> mTexCoords;
    std::vector<int> mSubMeshVertexStartIndices;
    GLuint mVao;
    GLuint mVbo[3];
    Sphere mBounds;
    bool mCreated;
    bool mChanged;

  public:
    Mesh();
    Mesh(Guid id);
    ~Mesh();

    virtual void serialize(std::ostream &out) const override;
    virtual void deserialize(std::istream &in) override;
    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void load(const std::string &filename);
    void load(std::vector<float> vertices, std::vector<float> normals, std::vector<float> texCoords,
              std::vector<int> subMeshStartIndices);

    bool isCreated() const;
    bool isChanged() const;

    const std::vector<float> &getVertices() const;
    const std::vector<float> &getNormals() const;
    const std::vector<float> &getTexCoords() const;
    const std::vector<int> &getSubMeshStartIndices() const;
    int getSubMeshStartIndex(int subMeshIndex) const;
    int getSubMeshEndIndex(int subMeshIndex) const;
    int getSubMeshCount() const;
    Sphere getBounds() const;
    GLuint getNativeGraphicsVAO() const;

    void setVertices(const std::vector<float> &vertices);
    void setNormals(const std::vector<float> &normals);
    void setTexCoords(const std::vector<float> &texCoords);

    void create();
    void destroy();
    void writeMesh();

  private:
    void computeBoundingSphere();
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