#ifndef TERRAIN_H__
#define TERRAIN_H__

#include <string>
#include <vector>

#include "Component.h"
#include "../core/Rect.h"

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

#include <vector>

namespace PhysicsEngine
{
struct TerrainChunk
{
    int mStart;
    int mEnd;
    Rect mRect;
    bool mEnabled;
};

class Terrain : public Component
{
    private:
        glm::ivec2 mChunkSize;
        TerrainChunk mTerrainChunks[9];
        Guid mMaterialId;

        std::vector<float> mVertices;
        std::vector<float> mNormals;
        std::vector<float> mTexCoords;

        unsigned int mVao;
        unsigned int mVbo[3];

        bool mCreated;
        bool mChanged;
        bool mMaterialChanged;

    public:
        float mScale;
        float mAmplitude;
        float mOffsetX;
        float mOffsetZ;
        //int mOctaves;
        //float mLacunarity;
        //float mGain;
        //float mOffset;
        Guid mCameraTransformId;

    public:
        Terrain(World *world);
        Terrain(World *world, const Guid &id);
        ~Terrain();

        virtual void serialize(YAML::Node& out) const override;
        virtual void deserialize(const YAML::Node& in) override;

        virtual int getType() const override;
        virtual std::string getObjectName() const override;

        void generateTerrain();
        void regenerateTerrain();
        void refine(int level);

        std::vector<float> getVertices() const;
        std::vector<float> getNormals() const;
        std::vector<float> getTexCoords() const;

        unsigned int getNativeGraphicsVAO() const;

        void setMaterial(Guid materialId);
        Guid getMaterial() const;

        bool isCreated() const;
        bool isChunkEnabled(int chunk) const;
        void enableChunk(int chunk);
        void disableChunk(int chunk);

        size_t getChunkStart(int chunk) const;
        size_t getChunkSize(int chunk) const;
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