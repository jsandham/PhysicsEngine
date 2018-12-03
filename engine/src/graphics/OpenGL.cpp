#include <iostream>

#include "../../include/graphics/OpenGL.h"

#include "../../include/core/Log.h"

using namespace PhysicsEngine;

GLenum OpenGL::getTextureFormat(TextureFormat format)
{
	GLenum openglFormat = GL_DEPTH_COMPONENT;

	switch (format)
	{
	case Depth:
		openglFormat = GL_DEPTH_COMPONENT;
		break;
	case RG:
		openglFormat = GL_RG;
		break;
	case RGB:
		openglFormat = GL_RGB;
		break;
	case RGBA:
		openglFormat = GL_RGBA;
		break;
	default:
		std::cout << "OpengGL: Invalid texture format" << std::endl;
	}

	return openglFormat;
}

void OpenGL::checkError()
{
	GLenum error;
	while ((error = glGetError()) != GL_NO_ERROR){
		std::cout << "OpenGL: Renderer failed with error code: " << error << std::endl;
	}
}

void OpenGL::enableDepthTest()
{
	glEnable(GL_DEPTH_TEST);
}

void OpenGL::enableCubemaps()
{
	glEnable(GL_TEXTURE_CUBE_MAP);
}

void OpenGL::enablePoints()
{
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);
}

void OpenGL::setViewport(int x, int y, int width, int height)
{
	glViewport(x, y, width, height);
}

void OpenGL::clearColorBuffer(glm::vec4 value)
{
	glClearColor(value.x, value.y, value.z, value.w);
	glClear(GL_COLOR_BUFFER_BIT);
}

void OpenGL::clearDepthBuffer(float value)
{
	glClearDepth(value);
	glClear(GL_DEPTH_BUFFER_BIT);
}