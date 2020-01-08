#ifndef __FRAMEBUFFER_DATA_H__
#define __FRAMEBUFFER_DATA_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/Shader.h"

namespace PhysicsEngine
{
	struct ScreenData // ScreenQuadData? RendererData? GraphicsScreenData? GraphicsRenderData?
	{
		Shader positionAndNormalsShader;  // whats a good name for this shader which fills depth, normals, and position? geometryShader? forwardGbufferShader?
		Shader testShader;

		// ssao fbo
		GLuint ssaoFBO;
		GLuint ssaoColor;
		Shader ssaoShader;

		// quad
		GLuint quadVAO;
		GLuint quadVBO;
		Shader quadShader;
	};
}

#endif
