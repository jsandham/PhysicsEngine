#ifndef __RENDEROBJECT_H__
#define __RENDEROBJECT_H__

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../core/Sphere.h"

namespace PhysicsEngine
{
	typedef struct RenderObject
	{
		Guid id;
		int start; // start index in vbo
		int size;  // size of vbo
		int transformIndex;
		int materialIndex;
		int shaderIndex;

		GLuint vao;

		GLint mainTexture;
		GLint normalMap;
		GLint specularMap;

		Sphere boundingSphere;

		glm::mat4 model;

		bool culled;
	}RenderObject;
}

#endif