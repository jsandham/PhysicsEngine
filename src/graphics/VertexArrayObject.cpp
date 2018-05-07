#include "VertexArrayObject.h"
#include <iostream>

using namespace PhysicsEngine;

VertexArrayObject::VertexArrayObject()
{
	drawMode = GL_TRIANGLES;
}

VertexArrayObject::~VertexArrayObject()
{
}

void VertexArrayObject::generate()
{
	glGenVertexArrays(1, &handle);
}

void VertexArrayObject::destroy()
{
	glDeleteVertexArrays(1, &handle);
}

void VertexArrayObject::bind()
{
	glBindVertexArray(handle);
}

void VertexArrayObject::unbind()
{
	glBindVertexArray(0);
}

void VertexArrayObject::setLayout(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, GLuint offset)
{
	glEnableVertexAttribArray(index);
	glVertexAttribPointer(index, size, type, normalized, stride, (GLvoid*)offset);
}

void VertexArrayObject::setDrawMode(GLenum mode)
{
	drawMode = mode;
}

void VertexArrayObject::draw(int count)
{
	glDrawArrays(drawMode, 0, count);
}
