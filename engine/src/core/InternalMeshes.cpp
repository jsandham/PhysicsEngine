#include "../../include/core/InternalMeshes.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

const std::vector<float> InternalMeshes::sphereVertices =
#include "meshes/sphere.vert"
;

const std::vector<float> InternalMeshes::sphereNormals =
#include "meshes/sphere.norm"
;

const std::vector<float> InternalMeshes::sphereTexCoords =
#include "meshes/sphere.texc"
;

const std::vector<float> InternalMeshes::cubeVertices =
#include "meshes/cube.vert"
;

const std::vector<float> InternalMeshes::cubeNormals =
#include "meshes/cube.norm"
;

const std::vector<float> InternalMeshes::cubeTexCoords =
#include "meshes/cube.texc"
;

const std::vector<float> InternalMeshes::planeVertices =
#include "meshes/plane.vert"
;

const std::vector<float> InternalMeshes::planeNormals =
#include "meshes/plane.norm"
;

const std::vector<float> InternalMeshes::planeTexCoords =
#include "meshes/plane.texc"
;

const std::vector<int> InternalMeshes::sphereSubMeshStartIndicies = { 0, 540 };
const std::vector<int> InternalMeshes::cubeSubMeshStartIndicies = { 0, 108 };
const std::vector<int> InternalMeshes::planeSubMeshStartIndicies = { 0, 18 };

const std::string InternalMeshes::sphereMeshName = "Sphere";
const std::string InternalMeshes::cubeMeshName = "Cube";
const std::string InternalMeshes::planeMeshName = "Plane";

Guid InternalMeshes::loadInternalMesh(World *world, const std::string &name,
                                      const std::vector<float> &vertices, const std::vector<float> &normals,
                                      const std::vector<float> &texCoords, const std::vector<int> &startIndices)
{
    PhysicsEngine::Mesh *mesh = world->createAsset<PhysicsEngine::Mesh>();
    if (mesh != nullptr)
    {
        mesh->load(vertices, normals, texCoords, startIndices);
        mesh->setName(name);
        return mesh->getId();
    }
    
    return Guid::INVALID;
}