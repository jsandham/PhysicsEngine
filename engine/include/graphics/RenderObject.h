#ifndef __RENDEROBJECT_H__
#define __RENDEROBJECT_H__

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	typedef struct RenderObject
	{
		bool culled;
		int start; // start index in vbo
		int size;  // size of vbo
		int transformIndex;
		int materialIndex;
		int shaderIndex;
		Sphere boundingSphere;

		glm::mat4 model;
	}RenderObject;
}

#endif