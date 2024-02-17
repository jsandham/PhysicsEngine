#ifndef TERRAIN_H__
#define TERRAIN_H__

#include "../core/glm.h"
#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/Rect.h"
#include "../core/Sphere.h"

#include "../graphics/MeshHandle.h"

#include "ComponentEnums.h"

namespace PhysicsEngine
{
struct TerrainChunk
{
    Rect mRect;
    int mStart;
    int mEnd;
    bool mEnabled;
};

struct TerrainCoverMesh
{
    Guid mMeshId;
    Guid mMaterialIds[8];

    int mMaterialCount;
};

class World;

class Terrain
{
  private:
    Guid mGuid;
    Id mId;
    Guid mEntityGuid;

    World *mWorld;

    TerrainChunk mTerrainChunks[81];
    TerrainCoverMesh mGrassMeshes[8];
    TerrainCoverMesh mTreeMeshes[8];
    Guid mMaterialId;

    glm::vec2 mChunkSize;
    glm::ivec2 mChunkResolution;

    std::vector<float> mVertices;
    std::vector<float> mNormals;
    std::vector<float> mTexCoords;

    std::vector<float> mPlaneVertices;
    std::vector<float> mPlaneTexCoords;

    MeshHandle *mHandle;

    VertexBuffer *mVertexBuffer;
    VertexBuffer *mNormalBuffer;
    VertexBuffer *mTexCoordsBuffer;

    int mTotalChunkCount;

    bool mCreated;
    bool mChanged;
    bool mMaterialChanged;
    bool mGrassMeshChanged;
    bool mTreeMeshChanged;

  public:
     HideFlag mHide;
     bool mEnabled;

    float mMaxViewDistance;

    float mScale;
    float mAmplitude;
    float mOffsetX;
    float mOffsetZ;

    int mGrassMeshCount;
    int mTreeMeshCount;
    Guid mCameraTransformId;

  public:
    Terrain(World *world, const Id &id);
    Terrain(World *world, const Guid &guid, const Id &id);
    ~Terrain();
    Terrain &operator=(Terrain &&other);

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getEntityGuid() const;
    Guid getGuid() const;
    Id getId() const;

    void generateTerrain();
    void regenerateTerrain();
    void updateTerrainHeight(float dx = 0.0f, float dz = 0.0f);

    std::vector<float> getVertices() const;
    std::vector<float> getNormals() const;
    std::vector<float> getTexCoords() const;

    MeshHandle *getNativeGraphicsHandle() const;

    void setMaterial(Guid materialId);
    void setGrassMesh(Guid meshId, int index);
    void setTreeMesh(Guid meshId, int index);
    void setGrassMaterial(Guid materialId, int meshIndex, int materialIndex);
    void setTreeMaterial(Guid materialId, int meshIndex, int materialIndex);
    Guid getMaterial() const;
    Guid getGrassMesh(int index) const;
    Guid getTreeMesh(int index) const;
    Guid getGrassMesh(int meshIndex, int materialIndex) const;
    Guid getTreeMesh(int meshIndex, int materialIndex) const;

    bool isCreated() const;
    bool isChunkEnabled(int chunk) const;
    void enableChunk(int chunk);
    void disableChunk(int chunk);

    size_t getChunkStart(int chunk) const;
    size_t getChunkSize(int chunk) const;
    Sphere getChunkBounds(int chunk) const;
    Rect getChunkRect(int chunk) const;
    Rect getCentreChunkRect() const;

    int getTotalChunkCount() const;

    template <typename T> T *getComponent() const
    {
        return mWorld->getActiveScene()->getComponent<T>(mEntityGuid);
    }

  private:
    friend class Scene;
};
} // namespace PhysicsEngine

#endif