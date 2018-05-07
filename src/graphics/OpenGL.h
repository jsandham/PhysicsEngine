#ifndef __OPENGL_H__
#define __OPENGL_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "Texture.h"
#include "Color.h"

#include "../glm/glm.hpp"

namespace PhysicsEngine
{
	class OpenGL
	{
		public:
			static GLenum getTextureFormat(TextureFormat format);
			static void checkError();
			static void enableDepthTest();
			static void enableCubemaps();
			static void enablePoints();
			static void setViewport(int x, int y, int width, int height);
			static void clearColorBuffer(glm::vec4 value);
			static void clearDepthBuffer(float value);
	};
}

#endif