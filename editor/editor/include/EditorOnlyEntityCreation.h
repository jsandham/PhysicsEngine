#ifndef __EDITOR_ONLY_ENTITY_CREATION_H__
#define __EDITOR_ONLY_ENTITY_CREATION_H__

#include <vector>
#include <set>

#include "core/Guid.h"
#include "core/World.h"

#include "components/Camera.h"
#include "components/Transform.h"

namespace PhysicsEditor
{
	PhysicsEngine::Camera* createEditorCamera(PhysicsEngine::World* world, std::set<PhysicsEngine::Guid>& editorOnlyIds);
	PhysicsEngine::Transform* createEditorTransformGizmo(PhysicsEngine::World* world, std::set<PhysicsEngine::Guid>& editorOnlyIds);
	PhysicsEngine::Transform* createEditorLightGizmo(PhysicsEngine::World* world, std::set<PhysicsEngine::Guid>& editorOnlyIds);
}

#endif
