#ifndef PHYSICS_H__
#define PHYSICS_H__

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

#include "../components/SphereCollider.h"

#include "OctTree.h"

namespace PhysicsEngine
{
class RaycastHit
{
};

class Physics
{
  private:
    static std::vector<SphereCollider *> colliders;
    // static Octtree tree;

  public:
    static float gravity;
    static float timestep;

  public:
    // static void init(Bounds bounds, int depth);
    // static void update(std::vector<Collider*> colliders);

    static bool raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance);
    static bool linecast(glm::vec3 start, glm::vec3 end);
    static bool spherecast(glm::vec3 centre, float radius, glm::vec3 direction);
    static bool boxcast(glm::vec3 centre, glm::vec3 size, glm::vec3 direction);

    static bool raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance, SphereCollider *collider);
    static bool linecast(glm::vec3 start, glm::vec3 end, SphereCollider *collider);
    static bool spherecast(glm::vec3 centre, float radius, glm::vec3 direction, SphereCollider *collider);
    static bool boxcast(glm::vec3 centre, glm::vec3 size, glm::vec3 direction, SphereCollider *collider);

    static std::vector<SphereCollider *> raycastAll(glm::vec3 origin, glm::vec3 direction, float maxDistance);
    static std::vector<SphereCollider *> linecastAll(glm::vec3 start, glm::vec3 end);
    static std::vector<SphereCollider *> spherecastAll(glm::vec3 centre, float radius, glm::vec3 direction);
    static std::vector<SphereCollider *> boxcastAll(glm::vec3 centre, glm::vec3 size, glm::vec3 direction);

    static Octtree getOcttree();
};
} // namespace PhysicsEngine

#endif