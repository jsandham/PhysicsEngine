#ifndef __GRAPHICS_TARGETS_H__
#define __GRAPHICS_TARGETS_H__	

#include "Graphics.h"

namespace PhysicsEngine
{
	typedef struct GraphicsTargets
	{
		GLint mColor;
		GLint mColorPicking;
		GLint mDepth;
		GLint mNormals;
		GLint mPosition;
		GLint mOverdraw;
		GLint mSsao;
		GLint mCascades;
		
		GraphicsTargets()
		{
			mColor = -1;
			mColorPicking = -1;
			mDepth = -1;
			mNormals = -1;
			mPosition = -1;
			mOverdraw = -1;
			mSsao = -1;
			mCascades = -1;
		}
	}GraphicsTargets;
}

#endif
