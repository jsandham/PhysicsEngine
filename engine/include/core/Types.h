#ifndef TYPES_H__
#define TYPES_H__

namespace PhysicsEngine
{
constexpr int INVALID_TYPE = -1;
constexpr int MIN_INTERNAL_ASSET = 1000;
constexpr int MAX_INTERNAL_ASSET = 1999;
constexpr int MIN_INTERNAL_COMPONENT = 2000;
constexpr int MAX_INTERNAL_COMPONENT = 2999;
constexpr int MIN_INTERNAL_SYSTEM = 3000;
constexpr int MAX_INTERNAL_SYSTEM = 3999;

constexpr int ENTITY_TYPE = 0;

constexpr int SHADER_TYPE = 1000;
constexpr int TEXTURE2D_TYPE = 1001;
constexpr int TEXTURE3D_TYPE = 1002;
constexpr int CUBEMAP_TYPE = 1003;
constexpr int MATERIAL_TYPE = 1004;
constexpr int MESH_TYPE = 1005;
constexpr int FONT_TYPE = 1006;

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

constexpr int RENDERSYSTEM_TYPE = 3000;
constexpr int PHYSICSSYSTEM_TYPE = 3001;
constexpr int CLEANUPSYSTEM_TYPE = 3002;
constexpr int DEBUGSYSTEM_TYPE = 3003;
constexpr int GIZMOSYSTEM_TYPE = 3004;
} // namespace PhysicsEngine

#endif