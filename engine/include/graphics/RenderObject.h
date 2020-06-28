#ifndef __RENDEROBJECT_H__
#define __RENDEROBJECT_H__

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../core/Sphere.h"
#include "../core/Color.h"

namespace PhysicsEngine
{
	typedef struct RenderObject
	{
		glm::mat4 model;
		Sphere boundingSphere;
		Guid meshRendererId;
		int meshRendererIndex;
		int materialIndex;
		int shaderIndex;
		int subMeshIndex;
		int start; // start index in vbo
		int size;  // size of vbo
		int vao;
		bool culled;
	}RenderObject;
}

#endif