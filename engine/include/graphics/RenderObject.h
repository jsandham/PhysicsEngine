#ifndef RENDEROBJECT_H__
#define RENDEROBJECT_H__

#include <cstdint>

namespace PhysicsEngine
{
    enum class DrawCallFlags
    {
        Indexed = 1,
        Instanced = 2,
        Batched = 4,
        Terrain = 8
    };

    class DrawCallCommand
    {
      private:
        uint64_t mDrawCallCode;

      public:
        DrawCallCommand();
        DrawCallCommand(uint64_t drawCallCode);

        uint64_t getCode() const;

        void generateTerrainDrawCall(int materialIndex, int terrainIndex, int chunk, int flags);
        void generateDrawCall(int materialIndex, int meshIndex, int subMesh, int depth, int flags);
        
        uint16_t getMaterialIndex() const;
        uint16_t getMeshIndex() const;
        uint8_t getSubMesh() const;
        uint32_t getDepth() const;
        uint8_t getFlags() const;

        void markDrawCallAsInstanced();
        void markDrawCallAsBatched();
        void markDrawCallAsIndexed();
        void markDrawCallAsTerrain();

        bool isInstanced() const;
        bool isBatched() const;
        bool isIndexed() const;
        bool isTerrain() const; 
    };

} // namespace PhysicsEngine
#endif