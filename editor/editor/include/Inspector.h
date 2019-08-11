#ifndef __INSPECTOR_H__
#define __INSPECTOR_H__

#include <vector>

#include "InspectorDrawer.h"

#include "core\Entity.h"

#include "components/Transform.h"
#include "components/Light.h"
#include "components/Camera.h"
#include "components/Rigidbody.h"
#include "components/MeshRenderer.h"
#include "components/LineRenderer.h"
#include "components/BoxCollider.h"
#include "components/SphereCollider.h"
#include "components/CapsuleCollider.h"
#include "components/MeshCollider.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
	class Inspector
	{
		private:
			std::vector<InspectorDrawer> drawers;
			Transform transform;
			Camera camera;
			Light light;
			Rigidbody rigidbody;
			MeshRenderer meshRenderer;
			LineRenderer lineRenderer;
			BoxCollider boxCollider;
			SphereCollider sphereCollider;
			CapsuleCollider capsuleCollider;
			MeshCollider meshCollider;

		public:
			Inspector();
			~Inspector();

			void render(Entity* entity, bool isOpenedThisFrame);





			bool BeginAddComponentDropdown(std::string filter, std::string& componentToAdd);
			void EndAddComponentDropdown();
	};
}

#endif