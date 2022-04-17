#ifndef TERRAIN_H__
#define TERRAIN_H__

#include "Component.h"
#include "../core/Rect.h"
#include "../core/Sphere.h"

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

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
    Id mMeshId;
    Id mMaterialIds[8];

    int mMaterialCount;
};

class Terrain : public Component
{
    private:
        TerrainChunk mTerrainChunks[81];
        TerrainCoverMesh mGrassMeshes[8];
        TerrainCoverMesh mTreeMeshes[8];
        Id mMaterialId;

        glm::vec2 mChunkSize;
        glm::ivec2 mChunkResolution;

        std::vector<float> mVertices;
        std::vector<float> mNormals;
        std::vector<float> mTexCoords;

        std::vector<float> mPlaneVertices;
        std::vector<float> mPlaneTexCoords;

        unsigned int mVao;
        unsigned int mVbo[3];

        int mTotalChunkCount;

        bool mCreated;
        bool mChanged;
        bool mMaterialChanged;
        bool mGrassMeshChanged;
        bool mTreeMeshChanged;

    public:
        float mMaxViewDistance;

        float mScale;
        float mAmplitude;
        float mOffsetX;
        float mOffsetZ;

        int mGrassMeshCount;
        int mTreeMeshCount;
        Id mCameraTransformId;

    public:
        Terrain(World *world);
        Terrain(World *world, Id id);
        ~Terrain();

        virtual void serialize(YAML::Node& out) const override;
        virtual void deserialize(const YAML::Node& in) override;

        virtual int getType() const override;
        virtual std::string getObjectName() const override;

        void generateTerrain();
        void regenerateTerrain();
        void updateTerrainHeight(float dx = 0.0f, float dz = 0.0f);

        std::vector<float> getVertices() const;
        std::vector<float> getNormals() const;
        std::vector<float> getTexCoords() const;

        unsigned int getNativeGraphicsVAO() const;

        void setMaterial(Id materialId);
        void setGrassMesh(Id meshId, int index);
        void setTreeMesh(Id meshId, int index);
        void setGrassMaterial(Id materialId, int meshIndex, int materialIndex);
        void setTreeMaterial(Id materialId, int meshIndex, int materialIndex);
        Id getMaterial() const;
        Id getGrassMesh(int index) const;
        Id getTreeMesh(int index) const;
        Id getGrassMesh(int meshIndex, int materialIndex) const;
        Id getTreeMesh(int meshIndex, int materialIndex) const;

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
};

template <> struct ComponentType<Terrain>
{
    static constexpr int type = PhysicsEngine::TERRAIN_TYPE;
};

template <> struct IsComponentInternal<Terrain>
{
    static constexpr bool value = true;
};
}

#endif