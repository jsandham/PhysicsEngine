#include "../../../../include/graphics/platform/directx/DirectXTextureHandle.h"

using namespace PhysicsEngine;

DirectXTextureHandle::DirectXTextureHandle()
{
}

DirectXTextureHandle::DirectXTextureHandle(int width, int height, TextureFormat format, TextureWrapMode wrapMode,
                             TextureFilterMode filterMode)
{
}

void DirectXTextureHandle::load(TextureFormat format,
	TextureWrapMode wrapMode,
	TextureFilterMode filterMode,
	int width,
	int height,
	const std::vector<unsigned char>& data)
{

}

void DirectXTextureHandle::update(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel)
{

}

void DirectXTextureHandle::readPixels(std::vector<unsigned char>& data)
{

}

void DirectXTextureHandle::writePixels(const std::vector<unsigned char>& data)
{

}

void DirectXTextureHandle::bind(unsigned int texUnit)
{

}

void DirectXTextureHandle::unbind(unsigned int texUnit)
{

}

void* DirectXTextureHandle::getHandle()
{
	return nullptr;// static_cast<void*>(&mHandle);
}