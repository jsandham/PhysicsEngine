#ifndef INTERNAL_MESHES_H__
#define INTERNAL_MESHES_H__

#include <vector>

#include "World.h"

namespace PhysicsEngine
{
class World;

class InternalMeshes
{
  public:
    static const std::vector<float> sphereVertices;
    static const std::vector<float> sphereNormals;
    static const std::vector<float> sphereTexCoords;
    static const std::vector<int> sphereSubMeshStartIndicies;

    static const std::vector<float> cubeVertices;
    static const std::vector<float> cubeNormals;
    static const std::vector<float> cubeTexCoords;
    static const std::vector<int> cubeSubMeshStartIndicies;

    static const std::vector<float> planeVertices;
    static const std::vector<float> planeNormals;
    static const std::vector<float> planeTexCoords;
    static const std::vector<int> planeSubMeshStartIndicies;

    static const std::string sphereMeshName;
    static const std::string cubeMeshName;
    static const std::string planeMeshName;

    enum class Mesh
    {
        Sphere,
        Cube,
        Plane
    };

    template<Mesh M>
    static Guid loadMesh(World* world)
    {
        return Guid::INVALID;
    }

    template<>
    static Guid loadMesh<Mesh::Sphere>(World* world)
    {
        return loadInternalMesh(world, sphereMeshName, sphereVertices,
            sphereNormals, sphereTexCoords, sphereSubMeshStartIndicies);
    }

    template<>
    static Guid loadMesh<Mesh::Cube>(World* world)
    {
        return loadInternalMesh(world, cubeMeshName, cubeVertices,
            cubeNormals, cubeTexCoords, cubeSubMeshStartIndicies);
    }

    template<>
    static Guid loadMesh<Mesh::Plane>(World* world)
    {
        return loadInternalMesh(world, planeMeshName, planeVertices,
            planeNormals, planeTexCoords, planeSubMeshStartIndicies);
    }

  private:
    static Guid loadInternalMesh(World *world, const std::string &name,
                                 const std::vector<float> &vertices, const std::vector<float> &normals,
                                 const std::vector<float> &texCoords, const std::vector<int> &startIndices);
};
} // namespace PhysicsEngine

#endif