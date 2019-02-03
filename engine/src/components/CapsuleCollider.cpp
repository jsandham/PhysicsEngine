#include "../../include/components/CapsuleCollider.h"

#include "../../include/core/Geometry.h"

using namespace PhysicsEngine;

CapsuleCollider::CapsuleCollider()
{
	
}

CapsuleCollider::CapsuleCollider(unsigned char* data)
{
	
}

CapsuleCollider::~CapsuleCollider()
{

}

void CapsuleCollider::load(CapsuleColliderData data)
{
	entityId = data.entityId;
	componentId = data.componentId;

	capsule = data.capsule;
}

bool CapsuleCollider::intersect(Bounds bounds)
{
	return Geometry::intersect(bounds, this->capsule);
}