#include "../../include/graphics/VertexBuffer.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/opengl/OpenGLVertexBuffer.h"
#include "../../include/graphics/platform/directx/DirectXVertexBuffer.h"

using namespace PhysicsEngine;

VertexBuffer::VertexBuffer()
{
}

VertexBuffer::~VertexBuffer()
{
}

VertexBuffer* VertexBuffer::create()
{
	switch (RenderContext::getRenderAPI())
	{
	case RenderAPI::OpenGL:
		return new OpenGLVertexBuffer();
	case RenderAPI::DirectX:
		return new DirectXVertexBuffer();
	}

	return nullptr;
}
