#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/directx/DirectXCubemapHandle.h"
#include "../../../../include/graphics/platform/directx/DirectXTextureHandle.h"

#include <glm/glm.hpp>

using namespace PhysicsEngine;

DirectXCubemapHandle::DirectXCubemapHandle(int width, TextureFormat format, TextureWrapMode wrapMode,
                                           TextureFilterMode filterMode)
    : CubemapHandle(width, format, wrapMode, filterMode)
{
}

DirectXCubemapHandle::~DirectXCubemapHandle()
{
}

void DirectXCubemapHandle::load(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                               const std::vector<unsigned char> &data)
{
}

void DirectXCubemapHandle::update(TextureWrapMode wrapMode, TextureFilterMode filterMode)
{
}

void DirectXCubemapHandle::readPixels(std::vector<unsigned char> &data)
{
}

void DirectXCubemapHandle::writePixels(const std::vector<unsigned char> &data)
{
}

void DirectXCubemapHandle::bind(unsigned int texUnit)
{
}

void DirectXCubemapHandle::unbind(unsigned int texUnit)
{
}

void *DirectXCubemapHandle::getHandle()
{
    return nullptr;
}