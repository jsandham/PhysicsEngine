#include "../include/EditorOnlyEntityCreation.h"

#include "core/InternalMeshes.h"
#include "core/InternalShaders.h"
#include "core/Material.h"
#include "core/Mesh.h"
#include "core/Shader.h"

using namespace PhysicsEditor;
using namespace PhysicsEngine;

Camera *PhysicsEditor::createEditorCamera(World *world, std::set<Guid> &editorOnlyIds)
{
    Entity *entity = world->createEntity();
    entity->mDoNotDestroy = true;

    Transform *transform = entity->addComponent<Transform>();
    Camera *camera = entity->addComponent<Camera>();

    // add entity id to editor only id list
    editorOnlyIds.insert(entity->getId());

    return camera;
}

Transform *PhysicsEditor::createEditorTransformGizmo(World *world, std::set<Guid> &editorOnlyIds)
{
    Entity *entity = world->createEntity();
    entity->mDoNotDestroy = true;

    Transform *transform = entity->addComponent<Transform>();
    transform->mPosition = glm::vec3(4, 4, 4);
    transform->mScale = glm::vec3(1, 1, 1);
    MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>();
    meshRenderer->mEnabled = false;

    Material *material = world->createAsset<Material>();
    Mesh *mesh = world->createAsset<Mesh>();
    Shader *shader = world->createAsset<Shader>();

    mesh->load(InternalMeshes::sphereVertices, InternalMeshes::sphereNormals, InternalMeshes::sphereTexCoords,
               InternalMeshes::sphereSubMeshStartIndicies);

    shader->load(InternalShaders::colorVertexShader, InternalShaders::colorFragmentShader, "");
    shader->compile();

    material->changeShader(shader->getId());
    material->onShaderChanged(world);
    material->setColor("color", Color::red);

    meshRenderer->setMesh(mesh->getId());
    meshRenderer->setMaterial(material->getId());

    // add entity id to editor only id list
    editorOnlyIds.insert(entity->getId());

    return transform;
}

Transform *PhysicsEditor::createEditorLightGizmo(PhysicsEngine::World *world,
                                                 std::set<PhysicsEngine::Guid> &editorOnlyIds)
{
    Entity *entity = world->createEntity();
    entity->mDoNotDestroy = true;

    Transform *transform = entity->addComponent<Transform>();
    transform->mPosition = glm::vec3(4, 4, 4);
    transform->mScale = glm::vec3(1, 1, 1);
    MeshRenderer *meshRenderer = entity->addComponent<MeshRenderer>();
    meshRenderer->mEnabled = false;

    Material *material = world->createAsset<Material>();
    Mesh *mesh = world->createAsset<Mesh>();
    Shader *shader = world->createAsset<Shader>();

    mesh->load(InternalMeshes::sphereVertices, InternalMeshes::sphereNormals, InternalMeshes::sphereTexCoords,
               InternalMeshes::sphereSubMeshStartIndicies);

    shader->load(InternalShaders::colorVertexShader, InternalShaders::colorFragmentShader, "");
    shader->compile();

    material->changeShader(shader->getId());
    material->onShaderChanged(world);
    material->setColor("color", Color::red);

    meshRenderer->setMesh(mesh->getId());
    meshRenderer->setMaterial(material->getId());

    // add entity id to editor only id list
    editorOnlyIds.insert(entity->getId());

    return transform;
}