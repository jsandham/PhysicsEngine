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
		Guid meshRendererId;
		Guid transformId;
		Guid meshId;
		Guid materialId;
		Guid shaderId;
		int meshRendererIndex;
		int transformIndex;
		int meshIndex;
		int materialIndex;
		int shaderIndex;
		int subMeshIndex;

		int start; // start index in vbo
		int size;  // size of vbo

		int vao;

		glm::mat4 model;

		Sphere boundingSphere;
		Color color;
		bool culled;
	}RenderObject;
}

#endif