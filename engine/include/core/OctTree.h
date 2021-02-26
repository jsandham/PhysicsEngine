#ifndef OCTTREE_H__
#define OCTTREE_H__

#include <vector>

#include "AABB.h"
#include "Capsule.h"
#include "Guid.h"
#include "Ray.h"
#include "Sphere.h"

#include "../components/Collider.h"

namespace PhysicsEngine
{
typedef struct Object1
{
    Guid mId;
    Sphere mSphere; // Primitive which sphere, bounds, capsule, triangle derive from?
} Object1;

typedef struct Node
{
    glm::vec3 mCentre;
    glm::vec3 mExtent;
    std::vector<Object1> mObjects;
} Node;

typedef struct Cell
{
    float mTx0;
    float mTy0;
    float mTz0;
    float mTx1;
    float mTy1;
    float mTz1;
    int mNodeIndex;
} Cell;

class Octtree
{
  private:
    int mMaxNumOfObjectsPerNode;
    int mDepth;
    AABB mBounds;
    std::vector<Node> mNodes;
    std::vector<float> mLines;

  public:
    Octtree();
    ~Octtree();

    void clear();
    void create(AABB bounds, int depth, int maxNumOfObjectsPerNode);
    void insert(Sphere sphere, Guid id);
    Object1 *intersect(Ray ray);

    int getDepth() const;
    AABB getBounds() const;
    std::vector<float> getLines() const;

  private:
    int firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm);
    int nextNode(float tx, int i, float ty, int j, float tz, int k);
};
} // namespace PhysicsEngine

#endif