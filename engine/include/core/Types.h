#ifndef __TYPES_H__
#define __TYPES_H__

namespace PhysicsEngine
{
	constexpr int INVALID_TYPE = -1;
	constexpr int MIN_INTERNAL_ASSET = 0;
	constexpr int MIN_INTERNAL_COMPONENT = 0;
	constexpr int MIN_INTERNAL_SYSTEM = 0;
	constexpr int MAX_INTERNAL_ASSET = 20;
	constexpr int MAX_INTERNAL_COMPONENT = 20;
	constexpr int MAX_INTERNAL_SYSTEM = 20;

	constexpr int ENTITY_TYPE = 0;

	constexpr int SHADER_TYPE = 0;
	constexpr int TEXTURE2D_TYPE = 1;
	constexpr int TEXTURE3D_TYPE = 2;
	constexpr int CUBEMAP_TYPE = 3;
	constexpr int MATERIAL_TYPE = 4;
	constexpr int MESH_TYPE = 5;
	constexpr int FONT_TYPE = 6;

	constexpr int TRANSFORM_TYPE = 0;
	constexpr int RIGIDBODY_TYPE = 1;
	constexpr int CAMERA_TYPE = 2;
	constexpr int MESHRENDERER_TYPE = 3;
	constexpr int LINERENDERER_TYPE = 4;
	constexpr int LIGHT_TYPE = 5;
	constexpr int BOXCOLLIDER_TYPE = 8;
	constexpr int SPHERECOLLIDER_TYPE = 9;
	constexpr int CAPSULECOLLIDER_TYPE = 10;
	constexpr int MESHCOLLIDER_TYPE = 15;

	constexpr int RENDERSYSTEM_TYPE = 0;
	constexpr int PHYSICSSYSTEM_TYPE = 1;
	constexpr int CLEANUPSYSTEM_TYPE = 2;
	constexpr int DEBUGSYSTEM_TYPE = 3;
}

#endif