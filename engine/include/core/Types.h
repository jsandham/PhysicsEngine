#ifndef TYPES_H__
#define TYPES_H__

namespace PhysicsEngine
{
// Types
constexpr int INVALID_TYPE = -1;
constexpr int MIN_INTERNAL_ASSET = 1000;
constexpr int MAX_INTERNAL_ASSET = 1999;
constexpr int MIN_INTERNAL_COMPONENT = 2000;
constexpr int MAX_INTERNAL_COMPONENT = 2999;
constexpr int MIN_INTERNAL_SYSTEM = 3000;
constexpr int MAX_INTERNAL_SYSTEM = 3999;

constexpr int SCENE_TYPE = 1;
constexpr int ENTITY_TYPE = 0;

constexpr int SHADER_TYPE = 1000;
constexpr int TEXTURE2D_TYPE = 1001;
constexpr int TEXTURE3D_TYPE = 1002;
constexpr int CUBEMAP_TYPE = 1003;
constexpr int MATERIAL_TYPE = 1004;
constexpr int MESH_TYPE = 1005;
constexpr int FONT_TYPE = 1006;
constexpr int SPRITE_TYPE = 1007;
constexpr int RENDER_TEXTURE_TYPE = 1008;

constexpr int TRANSFORM_TYPE = 2000;
constexpr int RIGIDBODY_TYPE = 2001;
constexpr int CAMERA_TYPE = 2002;
constexpr int MESHRENDERER_TYPE = 2003;
constexpr int LINERENDERER_TYPE = 2004;
constexpr int LIGHT_TYPE = 2005;
constexpr int BOXCOLLIDER_TYPE = 2006;
constexpr int SPHERECOLLIDER_TYPE = 2009;
constexpr int CAPSULECOLLIDER_TYPE = 2010;
constexpr int MESHCOLLIDER_TYPE = 2011;
constexpr int TERRAIN_TYPE = 2012;
constexpr int SPRITERENDERER_TYPE = 2013;

constexpr int RENDERSYSTEM_TYPE = 3000;
constexpr int PHYSICSSYSTEM_TYPE = 3001;
constexpr int CLEANUPSYSTEM_TYPE = 3002;
constexpr int DEBUGSYSTEM_TYPE = 3003;
constexpr int GIZMOSYSTEM_TYPE = 3004;
constexpr int FREELOOKCAMERASYSTEM_TYPE = 3005;

// Names
constexpr char SCENE_NAME[] = "Scene";
constexpr char ENTITY_NAME[] = "Entity";

constexpr char SHADER_NAME[] = "Shader";
constexpr char TEXTURE2D_NAME[] = "Texture2D";
constexpr char TEXTURE3D_NAME[] = "Texture3D";
constexpr char CUBEMAP_NAME[] = "Cubemap";
constexpr char MATERIAL_NAME[] = "Material";
constexpr char MESH_NAME[] = "Mesh";
constexpr char FONT_NAME[] = "Font";
constexpr char SPRITE_NAME[] = "Sprite";
constexpr char RENDER_TEXTURE_NAME[] = "RenderTexture";

constexpr char TRANSFORM_NAME[] = "Transform";
constexpr char RIGIDBODY_NAME[] = "Rigidbody";
constexpr char CAMERA_NAME[] = "Camera";
constexpr char MESHRENDERER_NAME[] = "MeshRenderer";
constexpr char LINERENDERER_NAME[] = "LineRenderer";
constexpr char LIGHT_NAME[] = "Light";
constexpr char BOXCOLLIDER_NAME[] = "BoxCollider";
constexpr char SPHERECOLLIDER_NAME[] = "SphereCollider";
constexpr char CAPSULECOLLIDER_NAME[] = "CapsuleCollider";
constexpr char MESHCOLLIDER_NAME[] = "MeshCollider";
constexpr char TERRAIN_NAME[] = "Terrain";
constexpr char SPRITERENDERER_NAME[] = "SpriteRenderer";

constexpr char RENDERSYSTEM_NAME[] = "RenderSystem";
constexpr char PHYSICSSYSTEM_NAME[] = "PhysicsSystem";
constexpr char CLEANUPSYSTEM_NAME[] = "CleanUpSystem";
constexpr char DEBUGSYSTEM_NAME[] = "DebugSystem";
constexpr char GIZMOSYSTEM_NAME[] = "GizmoSystem";
constexpr char FREELOOKCAMERASYSTEM_NAME[] = "FreeLookCameraSystem";

constexpr int isScene(int type)
{
    return type == SCENE_TYPE;
}

constexpr int isEntity(int type)
{
    return type == ENTITY_TYPE;
}

constexpr int isAsset(int type)
{
    return type >= MIN_INTERNAL_ASSET && type <= MAX_INTERNAL_ASSET;
}

constexpr int isComponent(int type)
{
    return type >= MIN_INTERNAL_COMPONENT && type <= MAX_INTERNAL_COMPONENT;
}

constexpr int isSystem(int type)
{
    return type >= MIN_INTERNAL_SYSTEM && type <= MAX_INTERNAL_SYSTEM;
}
} // namespace PhysicsEngine

#endif