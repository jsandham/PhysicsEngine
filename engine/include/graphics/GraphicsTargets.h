#ifndef __GRAPHICS_TARGETS_H__
#define __GRAPHICS_TARGETS_H__	

#include "Graphics.h"

namespace PhysicsEngine
{
	typedef struct GraphicsTargets
	{
		GLint color;
		GLint depth;
		GLint normals;
		GLint position;
		GLint overdraw;
		GLint ssao;
		GLint cascades;
		
		GraphicsTargets()
		{
			color = -1;
			depth = -1;
			normals = -1;
			position = -1;
			overdraw = -1;
			ssao = -1;
			cascades = -1;
		}
	}GraphicsTargets;
}

#endif
