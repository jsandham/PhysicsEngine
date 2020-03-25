#ifndef __FRAMEBUFFER_DATA_H__
#define __FRAMEBUFFER_DATA_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "../core/Shader.h"

namespace PhysicsEngine
{
	struct ScreenData // ScreenQuadData? RendererData? GraphicsScreenData? GraphicsRenderData?
	{
		Shader mPositionAndNormalsShader;  // whats a good name for this shader which fills depth, normals, and position? geometryShader? forwardGbufferShader?
		Shader mTestShader;

		// ssao fbo
		GLuint mSsaoFBO;
		GLuint mSsaoColor;
		Shader mSsaoShader;

		// quad
		GLuint mQuadVAO;
		GLuint mQuadVBO;
		Shader mQuadShader;
	};
}

#endif
