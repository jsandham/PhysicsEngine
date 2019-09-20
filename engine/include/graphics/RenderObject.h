#ifndef __RENDEROBJECT_H__
#define __RENDEROBJECT_H__

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	typedef struct RenderObject
	{
		Guid id;
		int start; // start index in vbo
		int size;  // size of vbo
		int transformIndex;
		int materialIndex;

		GLuint vao;

		GLuint shaders[10];
		GLint mainTexture;
		GLint normalMap;
		GLint specularMap;

		Sphere boundingSphere;

		glm::mat4 model;

		bool culled;
	}RenderObject;
}

#endif