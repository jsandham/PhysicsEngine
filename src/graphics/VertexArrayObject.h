#ifndef __VERTEXARRAYOBJECT_H__
#define __VERTEXARRAYOBJECT_H__

#include <GL/glew.h>

namespace PhysicsEngine
{

	class VertexArrayObject
	{
	private:
		GLenum drawMode;
		GLuint handle;

	public:
		VertexArrayObject();
		~VertexArrayObject();

		void generate();
		void destroy();
		void bind();
		void unbind();

		void setLayout(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, GLuint offset);
		void setDrawMode(GLenum mode);
		void draw(int count);
	};
}

#endif