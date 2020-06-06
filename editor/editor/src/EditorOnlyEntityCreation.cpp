#include "../include/EditorOnlyEntityCreation.h"

using namespace PhysicsEditor;
using namespace PhysicsEngine;

Camera* PhysicsEditor::createEditorCamera(World* world, std::set<Guid>& editorOnlyIds)
{
	Entity* entity = world->createEntity();
	entity->mDoNotDestroy = true;

	Transform* transform = entity->addComponent<Transform>(world);
	Camera* camera = entity->addComponent<Camera>(world);

	// add entity id to editor only id list
	editorOnlyIds.insert(entity->getId());

	return camera;
}

Transform* PhysicsEditor::createEditorTransformGizmo(World* world, std::set<Guid>& editorOnlyIds)
{
	Entity* entity = world->createEntity();
	entity->mDoNotDestroy = true;

	Transform* transform = entity->addComponent<Transform>(world);
	MeshRenderer* meshRenderer = entity->addComponent<MeshRenderer>(world);

	// add entity id to editor only id list
	editorOnlyIds.insert(entity->getId());

	return transform;
}