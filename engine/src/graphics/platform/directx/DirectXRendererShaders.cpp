#include "../../../../include/graphics/platform/directx/DirectXRendererShaders.h"

using namespace PhysicsEngine;

void DirectXRendererShaders::init_impl()
{

}

//void DirectXRendererShaders::compileSSAOShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileGeometryShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileDepthShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileDepthCubemapShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileScreenQuadShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileSpriteShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileGBufferShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileColorShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileColorInstancedShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileNormalShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileNormalInstancedShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compilePositionShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compilePositionInstancedShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileLinearDepthShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileLinearDepthInstancedShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileLineShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileGizmoShader_impl()
//{
//
//}
//
//void DirectXRendererShaders::compileGridShader_impl()
//{
//
//}

SSAOShader DirectXRendererShaders::getSSAOShader_impl()
{
	return mSSAOShader;
}

GeometryShader DirectXRendererShaders::getGeometryShader_impl()
{
	return mGeometryShader;
}

DepthShader DirectXRendererShaders::getDepthShader_impl()
{
	return mDepthShader;
}

DepthCubemapShader DirectXRendererShaders::getDepthCubemapShader_impl()
{
	return mDepthCubemapShader;
}

ScreenQuadShader DirectXRendererShaders::getScreenQuadShader_impl()
{
	return mScreenQuadShader;
}

SpriteShader DirectXRendererShaders::getSpriteShader_impl()
{
	return mSpriteShader;
}

GBufferShader DirectXRendererShaders::getGBufferShader_impl()
{
	return mGBufferShader;
}

ColorShader DirectXRendererShaders::getColorShader_impl()
{
	return mColorShader;
}

ColorInstancedShader DirectXRendererShaders::getColorInstancedShader_impl()
{
	return mColorInstancedShader;
}

NormalShader DirectXRendererShaders::getNormalShader_impl()
{
	return mNormalShader;
}

NormalInstancedShader DirectXRendererShaders::getNormalInstancedShader_impl()
{
	return mNormalInstancedShader;
}

PositionShader DirectXRendererShaders::getPositionShader_impl()
{
	return mPositionShader;
}

PositionInstancedShader DirectXRendererShaders::getPositionInstancedShader_impl()
{
	return mPositionInstancedShader;
}

LinearDepthShader DirectXRendererShaders::getLinearDepthShader_impl()
{
	return mLinearDepthShader;
}

LinearDepthInstancedShader DirectXRendererShaders::getLinearDepthInstancedShader_impl()
{
	return mLinearDepthInstancedShader;
}

LineShader DirectXRendererShaders::getLineShader_impl()
{
	return mLineShader;
}

GizmoShader DirectXRendererShaders::getGizmoShader_impl()
{
	return mGizmoShader;
}

GridShader DirectXRendererShaders::getGridShader_impl()
{
	return mGridShader;
}

std::string DirectXRendererShaders::getStandardVertexShader_impl()
{
	return "";
}

std::string DirectXRendererShaders::getStandardFragmentShader_impl()
{
	return "";
}