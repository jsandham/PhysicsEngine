#include <algorithm>
#include <iostream>
#include <stack>

#include "../../include/core/Intersect.h"
#include "../../include/core/Log.h"
#include "../../include/core/OctTree.h"

using namespace PhysicsEngine;

Octtree::Octtree()
{
}

Octtree::~Octtree()
{
}

void Octtree::clear()
{
    for (size_t i = 0; i < mNodes.size(); i++)
    {
        mNodes[i].mObjects.clear();
    }
}

void Octtree::create(AABB bounds, int depth, int maxNumOfObjectsPerNode)
{
    mBounds = bounds;
    mDepth = depth;
    mMaxNumOfObjectsPerNode = maxNumOfObjectsPerNode;

    // find total number of nodes in octtree corresponding to input max tree depth
    int d = 0;                 // depth
    int levelSize = 1;         // number of nodes only at depth d
    int totalSize = levelSize; // total number of nodes at all depths
    while (d < depth)
    {
        levelSize *= 8;
        d++;
        totalSize += levelSize;
    }

    mNodes.resize(totalSize);

    // initialize all octtree nodes to zero
    for (size_t i = 0; i < mNodes.size(); i++)
    {
        mNodes[i].mCentre = glm::vec3(0.0f, 0.0f, 0.0f);
        mNodes[i].mExtent = glm::vec3(0.0f, 0.0f, 0.0f);
        mNodes[i].mObjects.resize(maxNumOfObjectsPerNode);
        for (size_t j = 0; j < mNodes[i].mObjects.size(); j++)
        {
            mNodes[i].mObjects[j].mId = Guid::INVALID;
            mNodes[i].mObjects[j].mSphere.mRadius = 0.0f;
            mNodes[i].mObjects[j].mSphere.mCentre = glm::vec3(0.0f, 0.0f, 0.0f);
        }
    }

    std::cout << "Number of nodes allocated: " << mNodes.size() << " where each Node is: " << sizeof(mNodes[0])
              << " bytes" << std::endl;

    // for all nodes in octtree, determine centre and extents of node
    mNodes[0].mExtent = 0.5f * bounds.mSize;
    mNodes[0].mCentre = bounds.mCentre;

    std::cout << "root node centre: " << mNodes[0].mCentre.x << " " << mNodes[0].mCentre.y << " " << mNodes[0].mCentre.z
              << " root node extents: " << mNodes[0].mExtent.x << " " << mNodes[0].mExtent.y << " "
              << mNodes[0].mExtent.z << std::endl;

    std::stack<int> stack;
    stack.push(0);
    while (!stack.empty())
    {
        int currentIndex = stack.top();
        stack.pop();

        int maximumPossibleNodeIndex = 8 * currentIndex + 8;

        if (maximumPossibleNodeIndex < mNodes.size())
        {
            for (int i = -1; i <= 1; i += 2)
            {
                for (int j = -1; j <= 1; j += 2)
                {
                    for (int k = -1; k <= 1; k += 2)
                    {

                        glm::vec3 newExtent = 0.5f * mNodes[currentIndex].mExtent;
                        glm::vec3 newCentre;
                        newCentre.x = mNodes[currentIndex].mCentre.x + i * 0.5f * mNodes[currentIndex].mExtent.x;
                        newCentre.y = mNodes[currentIndex].mCentre.y + j * 0.5f * mNodes[currentIndex].mExtent.y;
                        newCentre.z = mNodes[currentIndex].mCentre.z + k * 0.5f * mNodes[currentIndex].mExtent.z;

                        int quadrant = 0;
                        for (int l = 0; l < 3; l++)
                        {
                            float delta = newCentre[l] - mNodes[currentIndex].mCentre[l];
                            if (delta > 0.0f)
                            {
                                quadrant |= (1 << l);
                            }
                        }

                        int index = 8 * currentIndex + quadrant + 1;
                        mNodes[index].mExtent = newExtent;
                        mNodes[index].mCentre = newCentre;

                        stack.push(index);
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < mNodes.size(); i++)
    {
        std::cout << "i: " << i << " " << mNodes[i].mCentre.x << " " << mNodes[i].mCentre.y << " "
                  << mNodes[i].mCentre.z << " " << mNodes[i].mExtent.x << " " << mNodes[i].mExtent.y << " "
                  << mNodes[i].mExtent.z << std::endl;
    }

    // create lines array
    mLines.resize(6 * 12 * mNodes.size());

    for (unsigned int i = 0; i < mNodes.size(); i++)
    {
        Node *node = &mNodes[i];

        // top
        mLines[6 * 12 * i] = node->mCentre.x - node->mExtent.x;
        mLines[6 * 12 * i + 1] = node->mCentre.y + node->mExtent.y;
        mLines[6 * 12 * i + 2] = node->mCentre.z + node->mExtent.z;
        mLines[6 * 12 * i + 3] = node->mCentre.x + node->mExtent.x;
        mLines[6 * 12 * i + 4] = node->mCentre.y + node->mExtent.y;
        mLines[6 * 12 * i + 5] = node->mCentre.z + node->mExtent.z;

        mLines[6 * 12 * i + 6] = node->mCentre.x + node->mExtent.x;
        mLines[6 * 12 * i + 7] = node->mCentre.y + node->mExtent.y;
        mLines[6 * 12 * i + 8] = node->mCentre.z + node->mExtent.z;
        mLines[6 * 12 * i + 9] = node->mCentre.x + node->mExtent.x;
        mLines[6 * 12 * i + 10] = node->mCentre.y - node->mExtent.y;
        mLines[6 * 12 * i + 11] = node->mCentre.z + node->mExtent.z;

        mLines[6 * 12 * i + 12] = node->mCentre.x + node->mExtent.x;
        mLines[6 * 12 * i + 13] = node->mCentre.y - node->mExtent.y;
        mLines[6 * 12 * i + 14] = node->mCentre.z + node->mExtent.z;
        mLines[6 * 12 * i + 15] = node->mCentre.x - node->mExtent.x;
        mLines[6 * 12 * i + 16] = node->mCentre.y - node->mExtent.y;
        mLines[6 * 12 * i + 17] = node->mCentre.z + node->mExtent.z;

        mLines[6 * 12 * i + 18] = node->mCentre.x - node->mExtent.x;
        mLines[6 * 12 * i + 19] = node->mCentre.y - node->mExtent.y;
        mLines[6 * 12 * i + 20] = node->mCentre.z + node->mExtent.z;
        mLines[6 * 12 * i + 21] = node->mCentre.x - node->mExtent.x;
        mLines[6 * 12 * i + 22] = node->mCentre.y + node->mExtent.y;
        mLines[6 * 12 * i + 23] = node->mCentre.z + node->mExtent.z;

        // bottom
        mLines[6 * 12 * i + 24] = node->mCentre.x - node->mExtent.x;
        mLines[6 * 12 * i + 25] = node->mCentre.y + node->mExtent.y;
        mLines[6 * 12 * i + 26] = node->mCentre.z - node->mExtent.z;
        mLines[6 * 12 * i + 27] = node->mCentre.x + node->mExtent.x;
        mLines[6 * 12 * i + 28] = node->mCentre.y + node->mExtent.y;
        mLines[6 * 12 * i + 29] = node->mCentre.z - node->mExtent.z;

        mLines[6 * 12 * i + 30] = node->mCentre.x + node->mExtent.x;
        mLines[6 * 12 * i + 31] = node->mCentre.y + node->mExtent.y;
        mLines[6 * 12 * i + 32] = node->mCentre.z - node->mExtent.z;
        mLines[6 * 12 * i + 33] = node->mCentre.x + node->mExtent.x;
        mLines[6 * 12 * i + 34] = node->mCentre.y - node->mExtent.y;
        mLines[6 * 12 * i + 35] = node->mCentre.z - node->mExtent.z;

        mLines[6 * 12 * i + 36] = node->mCentre.x + node->mExtent.x;
        mLines[6 * 12 * i + 37] = node->mCentre.y - node->mExtent.y;
        mLines[6 * 12 * i + 38] = node->mCentre.z - node->mExtent.z;
        mLines[6 * 12 * i + 39] = node->mCentre.x - node->mExtent.x;
        mLines[6 * 12 * i + 40] = node->mCentre.y - node->mExtent.y;
        mLines[6 * 12 * i + 41] = node->mCentre.z - node->mExtent.z;

        mLines[6 * 12 * i + 42] = node->mCentre.x - node->mExtent.x;
        mLines[6 * 12 * i + 43] = node->mCentre.y - node->mExtent.y;
        mLines[6 * 12 * i + 44] = node->mCentre.z - node->mExtent.z;
        mLines[6 * 12 * i + 45] = node->mCentre.x - node->mExtent.x;
        mLines[6 * 12 * i + 46] = node->mCentre.y + node->mExtent.y;
        mLines[6 * 12 * i + 47] = node->mCentre.z - node->mExtent.z;

        // sides
        mLines[6 * 12 * i + 48] = node->mCentre.x - node->mExtent.x;
        mLines[6 * 12 * i + 49] = node->mCentre.y + node->mExtent.y;
        mLines[6 * 12 * i + 50] = node->mCentre.z + node->mExtent.z;
        mLines[6 * 12 * i + 51] = node->mCentre.x - node->mExtent.x;
        mLines[6 * 12 * i + 52] = node->mCentre.y + node->mExtent.y;
        mLines[6 * 12 * i + 53] = node->mCentre.z - node->mExtent.z;

        mLines[6 * 12 * i + 54] = node->mCentre.x + node->mExtent.x;
        mLines[6 * 12 * i + 55] = node->mCentre.y + node->mExtent.y;
        mLines[6 * 12 * i + 56] = node->mCentre.z + node->mExtent.z;
        mLines[6 * 12 * i + 57] = node->mCentre.x + node->mExtent.x;
        mLines[6 * 12 * i + 58] = node->mCentre.y + node->mExtent.y;
        mLines[6 * 12 * i + 59] = node->mCentre.z - node->mExtent.z;

        mLines[6 * 12 * i + 60] = node->mCentre.x + node->mExtent.x;
        mLines[6 * 12 * i + 61] = node->mCentre.y - node->mExtent.y;
        mLines[6 * 12 * i + 62] = node->mCentre.z + node->mExtent.z;
        mLines[6 * 12 * i + 63] = node->mCentre.x + node->mExtent.x;
        mLines[6 * 12 * i + 64] = node->mCentre.y - node->mExtent.y;
        mLines[6 * 12 * i + 65] = node->mCentre.z - node->mExtent.z;

        mLines[6 * 12 * i + 66] = node->mCentre.x - node->mExtent.x;
        mLines[6 * 12 * i + 67] = node->mCentre.y - node->mExtent.y;
        mLines[6 * 12 * i + 68] = node->mCentre.z + node->mExtent.z;
        mLines[6 * 12 * i + 69] = node->mCentre.x - node->mExtent.x;
        mLines[6 * 12 * i + 70] = node->mCentre.y - node->mExtent.y;
        mLines[6 * 12 * i + 71] = node->mCentre.z - node->mExtent.z;
    }

    std::cout << "lines count: " << mLines.size() << std::endl;

    // tempLines.resize(lines.size());
}

void Octtree::insert(Sphere sphere, Guid id)
{
    Object1 object;
    object.mSphere = sphere;
    object.mId = id;

    int currentDepth = 0;

    std::stack<int> stack;

    stack.push(0);
    while (!stack.empty())
    {
        int nodeIndex = stack.top();
        stack.pop();

        // find quadrant that completely contains the object
        bool straddle = false;
        int quadrant = 0;
        for (int i = 0; i < 3; i++)
        {
            float delta = sphere.mCentre[i] - mNodes[nodeIndex].mCentre[i];
            if (std::abs(delta) <= sphere.mRadius)
            {
                straddle = true;
                break;
            }

            if (delta > 0.0f)
            {
                quadrant |= (1 << i);
            }
        }

        if (!straddle && currentDepth < mDepth)
        {
            stack.push(8 * nodeIndex + quadrant + 1);
        }
        else
        {
            // insert object into current node
            mNodes[nodeIndex].mObjects.push_back(object);
        }

        currentDepth++;
    }
}

int Octtree::getDepth() const
{
    return mDepth;
}

AABB Octtree::getBounds() const
{
    return mBounds;
}

std::vector<float> Octtree::getLines() const
{
    return mLines;
}

// Ray octtree intersection as described in the paper
// "An Efficient Parametric Algorithm for Octree Traversal" by Revelles, Urena, & Lastra
// Object* Octtree::intersect(Ray ray)
// {
// 	test = 0;

// 	std::cout << "origin.x: " << ray.origin.x << " origin.y: " << ray.origin.y << " origin.z: " << ray.origin.z << "
// direction.x: " << ray.direction.x << " direction.y: " << ray.direction.y << " ray.direction.z: " << ray.direction.z
// << "  bounds: " << bounds.size.x << " " << bounds.size.y << " " << bounds.size.z << std::endl;

// 	unsigned int a = 0;
// 	if(ray.direction.x < 0.0f){
// 		// ray.origin.x = bounds.size.x - ray.origin.x;
// 		ray.origin.x = 2.0f * bounds.centre.x - ray.origin.x;
// 		ray.direction.x = -ray.direction.x;
// 		// a |= 4;
// 		a |= 1;
// 	}

// 	if(ray.direction.y < 0.0f){
// 		// ray.origin.y = bounds.size.y - ray.origin.y;
// 		ray.origin.y = 2.0f * bounds.centre.y - ray.origin.y;
// 		ray.direction.y = -ray.direction.y;
// 		a |= 2;
// 	}

// 	if(ray.direction.z < 0.0f){
// 		// ray.origin.z = bounds.size.z - ray.origin.z;
// 		ray.origin.z = 2.0f * bounds.centre.z - ray.origin.z;
// 		ray.direction.z = -ray.direction.z;
// 		// a |= 1;
// 		a |= 4;
// 	}

// 	std::cout << "origin.x: " << ray.origin.x << " origin.y: " << ray.origin.y << " origin.z: " << ray.origin.z << "
// direction.x: " << ray.direction.x << " direction.y: " << ray.direction.y << " ray.direction.z: " << ray.direction.z
// << "  bounds: " << bounds.size.x << " " << bounds.size.y << " " << bounds.size.z << std::endl;

// 	float xmin = nodes[0].centre.x - nodes[0].extent.x;
// 	float xmax = nodes[0].centre.x + nodes[0].extent.x;
// 	float ymin = nodes[0].centre.y - nodes[0].extent.y;
// 	float ymax = nodes[0].centre.y + nodes[0].extent.y;
// 	float zmin = nodes[0].centre.z - nodes[0].extent.z;
// 	float zmax = nodes[0].centre.z + nodes[0].extent.z;

// 	Cell rootCell;
// 	rootCell.tx0 = (xmin - ray.origin.x) / ray.direction.x;
// 	rootCell.tx1 = (xmax - ray.origin.x) / ray.direction.x;
// 	rootCell.ty0 = (ymin - ray.origin.y) / ray.direction.y;
// 	rootCell.ty1 = (ymax - ray.origin.y) / ray.direction.y;
// 	rootCell.tz0 = (zmin - ray.origin.z) / ray.direction.z;
// 	rootCell.tz1 = (zmax - ray.origin.z) / ray.direction.z;
// 	rootCell.nodeIndex = 0;

// 	std::stack<Cell> stack;
// 	stack.push(rootCell);
// 	while(!stack.empty()){
// 		Cell currCell = stack.top();
// 		stack.pop();

// 		float tx0 = currCell.tx0;
// 		float tx1 = currCell.tx1;
// 		float ty0 = currCell.ty0;
// 		float ty1 = currCell.ty1;
// 		float tz0 = currCell.tz0;
// 		float tz1 = currCell.tz1;
// 		int nodeIndex = currCell.nodeIndex;

// 		// tx1, ty1, and tz1 cannot be negative if the ray intersects nodes[nodeIndex]
// 		if(tx1 < 0.0f || ty1 < 0.0f || tz1 < 0.0f){
// 			continue;
// 		}

// 		//std::cout << "Looking at node " << nodeIndex << " adjusted to: " << testNodeIndex << " for objects: " <<
// nodes[testNodeIndex].objects.size() << "(node is a child: " << (8*nodeIndex + 8 >= nodes.size()) << ")" << " tx0: "
// << tx0 << " ty0: " << ty0 << " tz0: " << tz0 << "  tx1: " << tx1 << " ty1: " << ty1 << " tz1: " << tz1 << "  txm: "
// << (0.5f * (tx0 + tx1)) << " tym: " << (0.5f * (ty0 + ty1)) << " tzm: " << (0.5f * (tz0 + tz1)) << std::endl;

// 		// ray intersects node nodes[nodeIndex] therefore add all objects in this node to search list
// 		//test += nodes[testNodeIndex].objects.size();

// 		// if node is a child node then dont try and look for lower children
// 		if(8*nodeIndex + 8 >= nodes.size()){
// 			//std::cout << "node index: " << nodeIndex << " is a child" << std::endl;
// 			continue;
// 		}

// 		float txm = 0.5f * (tx0 + tx1);
// 		float tym = 0.5f * (ty0 + ty1);
// 		float tzm = 0.5f * (tz0 + tz1);

// 		// find first node
// 		int localNodeIndex = firstNode(tx0, ty0, tz0, txm, tym, tzm);

// 		while(localNodeIndex < 8){
// 			std::cout << "first local node index: " << localNodeIndex << " direction corrected first local node index: "
// << (localNodeIndex ^ a) << std::endl; 			stack.push(8*nodeIndex + (localNodeIndex ^ a) + 1);
// 			//stack.push(8*nodeIndex + localNodeIndex + 1);
// 			// testStack.push(8*testNodeIndex + (localNodeIndex ^ a) + 1);

// 			switch(localNodeIndex)
// 			{
// 				case 0:
// 					// localNodeIndex = nextNode(txm, 4, tym, 2, tzm, 1);
// 					localNodeIndex = nextNode(txm, 1, tym, 2, tzm, 4);
// 					break;
// 				case 1:
// 					// localNodeIndex = nextNode(txm, 5, tym, 3, tz1, 8);
// 					localNodeIndex = nextNode(tx1, 8, tym, 3, tzm, 5);
// 					break;
// 				case 2:
// 					// localNodeIndex = nextNode(txm, 6, ty1, 8, tzm, 3);
// 					localNodeIndex = nextNode(txm, 3, ty1, 8, tzm, 6);
// 					break;
// 				case 3:
// 					// localNodeIndex = nextNode(txm, 7, ty1, 8, tz1, 8);
// 					localNodeIndex = nextNode(tx1, 8, ty1, 8, tzm, 7);
// 					break;
// 				case 4:
// 					// localNodeIndex = nextNode(tx1, 8, tym, 6, tzm, 5);
// 					localNodeIndex = nextNode(txm, 5, tym, 6, tz1, 8);
// 					break;
// 				case 5:
// 					// localNodeIndex = nextNode(tx1, 8, tym, 7, tz1, 8);
// 					localNodeIndex = nextNode(tx1, 8, tym, 7, tz1, 8);
// 					break;
// 				case 6:
// 					// localNodeIndex = nextNode(tx1, 8, ty1, 8, tzm, 7);
// 					localNodeIndex = nextNode(txm, 7, ty1, 8, tz1, 8);
// 					break;
// 				case 7:
// 					localNodeIndex = 8;
// 					break;
// 			}
// 		}
// 	}

// 	std::cout << "Number of objects that we need to test against: " << test << std::endl;

// 	return NULL;
// }

// Ray octtree intersection as described in the paper
// "An Efficient Parametric Algorithm for Octree Traversal" by Revelles, Urena, & Lastra
Object1 *Octtree::intersect(Ray ray)
{
    size_t test = 0;

    std::cout << "origin.x: " << ray.mOrigin.x << " origin.y: " << ray.mOrigin.y << " origin.z: " << ray.mOrigin.z
              << " direction.x: " << ray.mDirection.x << " direction.y: " << ray.mDirection.y
              << " ray.direction.z: " << ray.mDirection.z << "  bounds: " << mBounds.mSize.x << " " << mBounds.mSize.y
              << " " << mBounds.mSize.z << std::endl;

    unsigned int a = 0;
    if (ray.mDirection.x < 0.0f)
    {
        // ray.origin.x = bounds.size.x - ray.origin.x;
        ray.mOrigin.x = 2.0f * mBounds.mCentre.x - ray.mOrigin.x;
        ray.mDirection.x = -ray.mDirection.x;
        // a |= 4;
        a |= 1;
    }

    if (ray.mDirection.y < 0.0f)
    {
        // ray.origin.y = bounds.size.y - ray.origin.y;
        ray.mOrigin.y = 2.0f * mBounds.mCentre.y - ray.mOrigin.y;
        ray.mDirection.y = -ray.mDirection.y;
        a |= 2;
    }

    if (ray.mDirection.z < 0.0f)
    {
        // ray.origin.z = bounds.size.z - ray.origin.z;
        ray.mOrigin.z = 2.0f * mBounds.mCentre.z - ray.mOrigin.z;
        ray.mDirection.z = -ray.mDirection.z;
        // a |= 1;
        a |= 4;
    }

    std::stack<int> testStack; // contains the same indices as stack when the direction is positive. Instead of these
                               // two stack, use a single stack<Cell> stack
    testStack.push(0);
    std::stack<int> stack;
    stack.push(0);
    while (!stack.empty())
    {
        int nodeIndex = stack.top();
        stack.pop();

        int testNodeIndex = testStack.top();
        testStack.pop();

        float xmin = mNodes[nodeIndex].mCentre.x - mNodes[nodeIndex].mExtent.x;
        float xmax = mNodes[nodeIndex].mCentre.x + mNodes[nodeIndex].mExtent.x;
        float ymin = mNodes[nodeIndex].mCentre.y - mNodes[nodeIndex].mExtent.y;
        float ymax = mNodes[nodeIndex].mCentre.y + mNodes[nodeIndex].mExtent.y;
        float zmin = mNodes[nodeIndex].mCentre.z - mNodes[nodeIndex].mExtent.z;
        float zmax = mNodes[nodeIndex].mCentre.z + mNodes[nodeIndex].mExtent.z;

        float tx0 = (xmin - ray.mOrigin.x) / ray.mDirection.x;
        float tx1 = (xmax - ray.mOrigin.x) / ray.mDirection.x;
        float ty0 = (ymin - ray.mOrigin.y) / ray.mDirection.y;
        float ty1 = (ymax - ray.mOrigin.y) / ray.mDirection.y;
        float tz0 = (zmin - ray.mOrigin.z) / ray.mDirection.z;
        float tz1 = (zmax - ray.mOrigin.z) / ray.mDirection.z;

        // tx1, ty1, and tz1 cannot be negative if the ray intersects nodes[nodeIndex]
        if (tx1 < 0.0f || ty1 < 0.0f || tz1 < 0.0f)
        {
            continue;
        }

        std::cout << "Looking at node " << nodeIndex << " adjusted to: " << testNodeIndex
                  << " for objects: " << mNodes[testNodeIndex].mObjects.size()
                  << "(node is a child: " << (8 * nodeIndex + 8 >= mNodes.size()) << ")"
                  << " tx0: " << tx0 << " ty0: " << ty0 << " tz0: " << tz0 << "  tx1: " << tx1 << " ty1: " << ty1
                  << " tz1: " << tz1 << "  txm: " << (0.5f * (tx0 + tx1)) << " tym: " << (0.5f * (ty0 + ty1))
                  << " tzm: " << (0.5f * (tz0 + tz1)) << std::endl;

        // ray intersects node nodes[nodeIndex] therefore add all objects in this node to search list
        test += mNodes[testNodeIndex].mObjects.size();

        // if node is a child node then dont try and look for lower children
        if (8 * nodeIndex + 8 >= mNodes.size())
        {
            // std::cout << "node index: " << nodeIndex << " is a child" << std::endl;
            continue;
        }

        float txm = 0.5f * (tx0 + tx1);
        float tym = 0.5f * (ty0 + ty1);
        float tzm = 0.5f * (tz0 + tz1);

        // find first node
        int localNodeIndex = firstNode(tx0, ty0, tz0, txm, tym, tzm);

        while (localNodeIndex < 8)
        {
            int temp = (localNodeIndex ^ a);
            std::cout << "first local node index: " << localNodeIndex
                      << " direction corrected first local node index: " << temp << std::endl;
            // stack.push(8*nodeIndex + (localNodeIndex ^ a) + 1);
            stack.push(8 * nodeIndex + localNodeIndex + 1);
            testStack.push(8 * testNodeIndex + (localNodeIndex ^ a) + 1);

            switch (localNodeIndex)
            {
            case 0:
                // localNodeIndex = nextNode(txm, 4, tym, 2, tzm, 1);
                localNodeIndex = nextNode(txm, 1, tym, 2, tzm, 4);
                break;
            case 1:
                // localNodeIndex = nextNode(txm, 5, tym, 3, tz1, 8);
                localNodeIndex = nextNode(tx1, 8, tym, 3, tzm, 5);
                break;
            case 2:
                // localNodeIndex = nextNode(txm, 6, ty1, 8, tzm, 3);
                localNodeIndex = nextNode(txm, 3, ty1, 8, tzm, 6);
                break;
            case 3:
                // localNodeIndex = nextNode(txm, 7, ty1, 8, tz1, 8);
                localNodeIndex = nextNode(tx1, 8, ty1, 8, tzm, 7);
                break;
            case 4:
                // localNodeIndex = nextNode(tx1, 8, tym, 6, tzm, 5);
                localNodeIndex = nextNode(txm, 5, tym, 6, tz1, 8);
                break;
            case 5:
                // localNodeIndex = nextNode(tx1, 8, tym, 7, tz1, 8);
                localNodeIndex = nextNode(tx1, 8, tym, 7, tz1, 8);
                break;
            case 6:
                // localNodeIndex = nextNode(tx1, 8, ty1, 8, tzm, 7);
                localNodeIndex = nextNode(txm, 7, ty1, 8, tz1, 8);
                break;
            case 7:
                localNodeIndex = 8;
                break;
            }
        }
    }

    std::cout << "Number of objects that we need to test against: " << test << std::endl;

    return NULL;
}

// First node selection as described in the paper
// "An Efficient Parametric Algorithm for Octree Traversal" by Revelles, Urena, & Lastra
int Octtree::firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm)
{
    int index = 0;
    if (tx0 >= std::max(ty0, tz0))
    { // enters YZ plane
        std::cout << "first enters YZ plane" << std::endl;
        if (tym < tx0)
        {
            index = index | (1 << 1);
        }
        if (tzm < tx0)
        {
            index = index | (1 << 2);
        }
    }
    else if (ty0 >= std::max(tx0, tz0))
    { // enters XZ plane
        std::cout << "first enters XZ plane" << std::endl;
        if (txm < ty0)
        {
            index = index | (1 << 0);
        } // 0
        if (tzm < ty0)
        {
            index = index | (1 << 2);
        } // 1
    }
    else // enters XY plane
    {
        std::cout << "first enters XY plane" << std::endl;
        if (txm < tz0)
        {
            index = index | (1 << 0);
        }
        if (tym < tz0)
        {
            index = index | (1 << 1);
        }
    }

    return index;
}

// Next node selection as described in the paper
// "An Efficient Parametric Algorithm for Octree Traversal" by Revelles, Urena, & Lastra
int Octtree::nextNode(float tx, int i, float ty, int j, float tz, int k)
{
    if (tx < std::min(ty, tz))
    { // YZ plane
        std::cout << "next enters YZ plane" << std::endl;
        return i;
    }
    else if (ty < std::min(tx, tz))
    { // XZ plane
        std::cout << "next enters XZ plane" << std::endl;
        return j;
    }
    else
    { // XY plane
        std::cout << "next enters XY plane" << std::endl;
        return k;
    }
}

// size_t Octtree::test = 0;

// void Octtree::tempClear()
// {
// 	tempObjects.clear();
// }

// void Octtree::tempInsert(Sphere sphere, Guid id)
// {
// 	Object obj;
// 	obj.sphere = sphere;
// 	obj.id = id;

// 	tempObjects.push_back(obj);
// }

// Object* Octtree::tempIntersect(Ray ray)
// {
// 	for(int i = 0; i < tempObjects.size(); i++){
// 		if(Geometry::intersect(ray, tempObjects[i].sphere)){
// 			return &tempObjects[i];
// 		}
// 	}

// 	return NULL;
// }

// std::vector<float> Octtree::getLinesTemp()
// {
// 	// tempLines.clear();
// 	for(unsigned int i = 0; i < tempLines.size(); i++){
// 		tempLines[i] = 0.0f;
// 	}

// 	int index = 0;
// 	for(unsigned int i = 0; i < nodes.size(); i++){
// 		Node* node = &nodes[i];
// 		if(node->objects.size() > 0){
// 			//std::cout << "node index with objects: " << i << std::endl;

// 			// top
// 			tempLines[6*12*index] = node->centre.x - node->extent.x;
// 			tempLines[6*12*index + 1] = node->centre.y + node->extent.y;
// 			tempLines[6*12*index + 2] = node->centre.z + node->extent.z;
// 			tempLines[6*12*index + 3] = node->centre.x + node->extent.x;
// 			tempLines[6*12*index + 4] = node->centre.y + node->extent.y;
// 			tempLines[6*12*index + 5] = node->centre.z + node->extent.z;

// 			tempLines[6*12*index + 6] = node->centre.x + node->extent.x;
// 			tempLines[6*12*index + 7] = node->centre.y + node->extent.y;
// 			tempLines[6*12*index + 8] = node->centre.z + node->extent.z;
// 			tempLines[6*12*index + 9] = node->centre.x + node->extent.x;
// 			tempLines[6*12*index + 10] = node->centre.y - node->extent.y;
// 			tempLines[6*12*index + 11] = node->centre.z + node->extent.z;

// 			tempLines[6*12*index + 12] = node->centre.x + node->extent.x;
// 			tempLines[6*12*index + 13] = node->centre.y - node->extent.y;
// 			tempLines[6*12*index + 14] = node->centre.z + node->extent.z;
// 			tempLines[6*12*index + 15] = node->centre.x - node->extent.x;
// 			tempLines[6*12*index + 16] = node->centre.y - node->extent.y;
// 			tempLines[6*12*index + 17] = node->centre.z + node->extent.z;

// 			tempLines[6*12*index + 18] = node->centre.x - node->extent.x;
// 			tempLines[6*12*index + 19] = node->centre.y - node->extent.y;
// 			tempLines[6*12*index + 20] = node->centre.z + node->extent.z;
// 			tempLines[6*12*index + 21] = node->centre.x - node->extent.x;
// 			tempLines[6*12*index + 22] = node->centre.y + node->extent.y;
// 			tempLines[6*12*index + 23] = node->centre.z + node->extent.z;

// 			// bottom
// 			tempLines[6*12*index + 24] = node->centre.x - node->extent.x;
// 			tempLines[6*12*index + 25] = node->centre.y + node->extent.y;
// 			tempLines[6*12*index + 26] = node->centre.z - node->extent.z;
// 			tempLines[6*12*index + 27] = node->centre.x + node->extent.x;
// 			tempLines[6*12*index + 28] = node->centre.y + node->extent.y;
// 			tempLines[6*12*index + 29] = node->centre.z - node->extent.z;

// 			tempLines[6*12*index + 30] = node->centre.x + node->extent.x;
// 			tempLines[6*12*index + 31] = node->centre.y + node->extent.y;
// 			tempLines[6*12*index + 32] = node->centre.z - node->extent.z;
// 			tempLines[6*12*index + 33] = node->centre.x + node->extent.x;
// 			tempLines[6*12*index + 34] = node->centre.y - node->extent.y;
// 			tempLines[6*12*index + 35] = node->centre.z - node->extent.z;

// 			tempLines[6*12*index + 36] = node->centre.x + node->extent.x;
// 			tempLines[6*12*index + 37] = node->centre.y - node->extent.y;
// 			tempLines[6*12*index + 38] = node->centre.z - node->extent.z;
// 			tempLines[6*12*index + 39] = node->centre.x - node->extent.x;
// 			tempLines[6*12*index + 40] = node->centre.y - node->extent.y;
// 			tempLines[6*12*index + 41] = node->centre.z - node->extent.z;

// 			tempLines[6*12*index + 42] = node->centre.x - node->extent.x;
// 			tempLines[6*12*index + 43] = node->centre.y - node->extent.y;
// 			tempLines[6*12*index + 44] = node->centre.z - node->extent.z;
// 			tempLines[6*12*index + 45] = node->centre.x - node->extent.x;
// 			tempLines[6*12*index + 46] = node->centre.y + node->extent.y;
// 			tempLines[6*12*index + 47] = node->centre.z - node->extent.z;

// 			// sides
// 			tempLines[6*12*index + 48] = node->centre.x - node->extent.x;
// 			tempLines[6*12*index + 49] = node->centre.y + node->extent.y;
// 			tempLines[6*12*index + 50] = node->centre.z + node->extent.z;
// 			tempLines[6*12*index + 51] = node->centre.x - node->extent.x;
// 			tempLines[6*12*index + 52] = node->centre.y + node->extent.y;
// 			tempLines[6*12*index + 53] = node->centre.z - node->extent.z;

// 			tempLines[6*12*index + 54] = node->centre.x + node->extent.x;
// 			tempLines[6*12*index + 55] = node->centre.y + node->extent.y;
// 			tempLines[6*12*index + 56] = node->centre.z + node->extent.z;
// 			tempLines[6*12*index + 57] = node->centre.x + node->extent.x;
// 			tempLines[6*12*index + 58] = node->centre.y + node->extent.y;
// 			tempLines[6*12*index + 59] = node->centre.z - node->extent.z;

// 			tempLines[6*12*index + 60] = node->centre.x + node->extent.x;
// 			tempLines[6*12*index + 61] = node->centre.y - node->extent.y;
// 			tempLines[6*12*index + 62] = node->centre.z + node->extent.z;
// 			tempLines[6*12*index + 63] = node->centre.x + node->extent.x;
// 			tempLines[6*12*index + 64] = node->centre.y - node->extent.y;
// 			tempLines[6*12*index + 65] = node->centre.z - node->extent.z;

// 			tempLines[6*12*index + 66] = node->centre.x - node->extent.x;
// 			tempLines[6*12*index + 67] = node->centre.y - node->extent.y;
// 			tempLines[6*12*index + 68] = node->centre.z + node->extent.z;
// 			tempLines[6*12*index + 69] = node->centre.x - node->extent.x;
// 			tempLines[6*12*index + 70] = node->centre.y - node->extent.y;
// 			tempLines[6*12*index + 71] = node->centre.z - node->extent.z;

// 			index++;
// 		}
// 	}

// 	return tempLines;
// }

// Node::Node()
// {

// }

// Node::~Node()
// {

// }

// bool Node::containsAny()
// {
// 	return indices.size() > 0;
// }

// bool Node::contains(int index)
// {
// 	for (unsigned int i = 0; i < indices.size(); i++){
// 		if (indices[i] == index){
// 			return true;
// 		}
// 	}

// 	return false;
// }

// void Node::add(int index)
// {
// 	for (unsigned int i = 0; i < indices.size(); i++){
// 		if (indices[i] == index){
// 			return;
// 		}
// 	}

// 	indices.push_back(index);
// }

// void Node::clear()
// {
// 	indices.clear();
// }

// Octtree::Octtree()
// {

// }

// Octtree::Octtree(Bounds bounds, int depth)
// {
// 	allocate(bounds, depth);
// }

// Octtree::~Octtree()
// {

// }

// void Octtree::allocate(Bounds bounds, int depth)
// {
// 	this->bounds = bounds;

// 	int size = 1;
// 	int levelSize = 1;
// 	int d = 0;
// 	while (d < depth){
// 		levelSize *= 8;
// 		size += levelSize;
// 		d++;
// 	}

// 	nodes.resize(size);

// 	nodes[0].bounds = bounds;

// 	std::stack<int> stack;
// 	stack.push(0);
// 	while (!stack.empty()){
// 		int index = stack.top();
// 		stack.pop();

// 		if(8*index + 8 < nodes.size()){
// 			glm::vec3 extents = nodes[index].bounds.getExtents();

// 			for (int i = 1; i <= 8; i++){
// 				nodes[8 * index + i].bounds.size = extents;
// 			}

// 			nodes[8 * index + 1].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(extents.x, extents.y,
// extents.z); 			nodes[8 * index + 2].bounds.centre = nodes[index].bounds.centre + 0.5f * glm::vec3(extents.x,
// extents.y, -extents.z); 			nodes[8 * index + 3].bounds.centre = nodes[index].bounds.centre + 0.5f *
// glm::vec3(extents.x, -extents.y, extents.z); 			nodes[8 * index + 4].bounds.centre =
// nodes[index].bounds.centre + 0.5f * glm::vec3(-extents.x, extents.y, extents.z); 			nodes[8 * index +
// 5].bounds.centre = nodes[index].bounds.centre + 0.5f *
// glm::vec3(extents.x, -extents.y, -extents.z); 			nodes[8 * index + 6].bounds.centre = nodes[index].bounds.centre +
// 0.5f
// * glm::vec3(-extents.x, -extents.y, extents.z); 			nodes[8 * index + 7].bounds.centre = nodes[index].bounds.centre
// + 0.5f * glm::vec3(-extents.x, extents.y, -extents.z); 			nodes[8 * index + 8].bounds.centre =
// nodes[index].bounds.centre
// + 0.5f * glm::vec3(-extents.x, -extents.y, -extents.z);

// 			for (int i = 8 * index + 1; i <= 8 * index + 8; i++){
// 				stack.push(i);
// 			}
// 		}
// 	}
// }

// void Octtree::build(std::vector<Collider*> colliders)
// {
// 	this->colliders = colliders;

// 	//Log::Info("building octtree with %d colliders", colliders.size());

// 	for (unsigned int i = 0; i < colliders.size(); i++){

// 		std::stack<int> stack;
// 		stack.push(0);
// 		while (!stack.empty()){
// 			int index = stack.top();
// 			stack.pop();
// 			if (colliders[i]->intersect(nodes[index].bounds)){
// 				nodes[index].add(i);

// 				//Log::Info("collider %d intersected with node %d", i, index);

// 				if(8*index+8 < nodes.size()){
// 					for (int j = 8 * index + 1; j <= 8 * index + 8; j++){
// 						stack.push(j);
// 					}
// 				}
// 			}
// 		}
// 	}
// }

// std::vector<float> Octtree::getWireframe()
// {
// 	std::vector<float> vertices;
// 	for(unsigned int i = 0; i < nodes.size(); i++){
// 		Node* node = &nodes[i];

// 		if(node->containsAny()){
// 			glm::vec3 centre = node->bounds.centre;
// 			glm::vec3 extents = 0.5f * node->bounds.size;

// 			//Log::Info("extents: %f %f %f", extents.x, extents.y, extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			//
// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			//
// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y - extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x + extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z + extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z - extents.z);

// 			vertices.push_back(centre.x - extents.x);
// 			vertices.push_back(centre.y + extents.y);
// 			vertices.push_back(centre.z + extents.z);
// 		}
// 	}

// 	return vertices;
// }