#include "../../include/core/Physics.h"

using namespace PhysicsEngine;

float Physics::gravity = 9.81f;
float Physics::timestep = 0.01f;

std::vector<SphereCollider *> Physics::colliders;
// Octtree Physics::tree;

// void Physics::init(Bounds bounds, int depth)
// {
// 	tree.allocate(bounds, depth);
// }

// void Physics::update(std::vector<Collider*> colliders)
// {
// 	Physics::colliders = colliders;

// 	Physics::tree.build(colliders);
// }

bool Physics::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance)
{
    return true;
}

bool Physics::linecast(glm::vec3 start, glm::vec3 end)
{
    return true;
}

bool Physics::spherecast(glm::vec3 centre, float radius, glm::vec3 direction)
{
    return true;
}

bool Physics::boxcast(glm::vec3 centre, glm::vec3 size, glm::vec3 direction)
{
    return true;
}

bool Physics::raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance, SphereCollider *collider)
{
    return true;
}

bool Physics::linecast(glm::vec3 start, glm::vec3 end, SphereCollider *collider)
{
    return true;
}

bool Physics::spherecast(glm::vec3 centre, float radius, glm::vec3 direction, SphereCollider *collider)
{
    return true;
}

bool Physics::boxcast(glm::vec3 centre, glm::vec3 size, glm::vec3 direction, SphereCollider *collider)
{
    return true;
}

std::vector<SphereCollider *> Physics::raycastAll(glm::vec3 origin, glm::vec3 direction, float maxDistance)
{
    std::vector<SphereCollider *> hitColliders;

    return hitColliders;
}

std::vector<SphereCollider *> Physics::linecastAll(glm::vec3 start, glm::vec3 end)
{
    std::vector<SphereCollider *> hitColliders;

    return hitColliders;
}

std::vector<SphereCollider *> Physics::spherecastAll(glm::vec3 centre, float radius, glm::vec3 direction)
{
    std::vector<SphereCollider *> hitColliders;

    return hitColliders;
}

std::vector<SphereCollider *> Physics::boxcastAll(glm::vec3 centre, glm::vec3 size, glm::vec3 direction)
{
    std::vector<SphereCollider *> hitColliders;

    return hitColliders;
}

// Octtree Physics::getOcttree()
// {
// 	return Physics::tree;
// }