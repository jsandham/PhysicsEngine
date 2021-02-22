#ifndef __INTERNAL_MESHES_H__
#define __INTERNAL_MESHES_H__

#include <vector>

#include "Guid.h"
#include "Mesh.h"

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

    static const Guid sphereMeshId;
    static const Guid cubeMeshId;
    static const Guid planeMeshId;

    static const std::string sphereMeshName;
    static const std::string cubeMeshName;
    static const std::string planeMeshName;

    static Guid loadSphereMesh(World *world);
    static Guid loadCubeMesh(World *world);
    static Guid loadPlaneMesh(World *world);

  private:
    static Guid loadInternalMesh(World *world, const Guid meshId, const std::string &name,
                                 const std::vector<float> &vertices, const std::vector<float> &normals,
                                 const std::vector<float> &texCoords, const std::vector<int> &startIndices);
};
} // namespace PhysicsEngine

#endif