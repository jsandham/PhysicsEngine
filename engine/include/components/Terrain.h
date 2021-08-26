#ifndef TERRAIN_H__
#define TERRAIN_H__

#include <string>
#include <vector>

#include "Component.h"

#include <GL/glew.h>
#include <gl/gl.h>

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

namespace PhysicsEngine
{
/*enum class TerrainMode
{
    Triangular = 0,
    Voxel = 1
};

enum class VoxelFace
{
    Left = 0,
    Right = 1,
    Near = 2,
    Far = 3,
    Top = 4,
    Bottom = 5
};

struct Voxel
{
    char faces[6];

};

struct Chunk
{
    Voxel voxels[128*128*128];
    glm::ivec2 position;
};

class Terrain : public Component
{
    private:
        TerrainMode mMode;
        glm::ivec2 mSize;
        std::vector<float> mHeightMap;

        std::vector<glm::vec3> mVertices;
        std::vector<glm::vec3> mNormals;

        GLuint mVao;
        GLuint mVbo[3];
        bool mCreated;
        bool mChanged;

    public:
        Terrain();
        ~Terrain();

        virtual void serialize(std::ostream& out) const override;
        virtual void deserialize(std::istream& in) override;
        virtual void serialize(YAML::Node& out) const override;
        virtual void deserialize(const YAML::Node& in) override;

        virtual int getType() const override;
        virtual std::string getObjectName() const override;

        void regenerateTerrain();
        void refine(int level);

        GLuint getNativeGraphicsVAO() const;

        void create();
        void destroy();
};


template <> struct ComponentType<Terrain>
{
    static constexpr int type = PhysicsEngine::TERRAIN_TYPE;
};

template <> struct IsComponentInternal<Terrain>
{
    static constexpr bool value = true;
};*/
}

#endif