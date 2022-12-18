#ifndef DIRECTX_TEXTURE_HANDLE_H__
#define DIRECTX_TEXTURE_HANDLE_H__

#include "../../TextureHandle.h"

#include <windows.h>
#include <d3d11.h>

namespace PhysicsEngine
{
	class DirectXTextureHandle : public TextureHandle
	{
	public:
		D3D11_TEXTURE2D_DESC mDesc;
		ID3D11Texture2D* mTexture;
		ID3D11ShaderResourceView* mResourceView;
		ID3D11SamplerState* mSamplerState;
		D3D11_SAMPLER_DESC mSamplerDesc;

		DirectXTextureHandle();
        DirectXTextureHandle(int width, int height, TextureFormat format,
                                                   TextureWrapMode wrapMode, TextureFilterMode filterMode);
		void load(TextureFormat format,
			TextureWrapMode wrapMode,
			TextureFilterMode filterMode,
			int width,
			int height,
			const std::vector<unsigned char>& data) override;
        void update(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel) override;
		void readPixels(std::vector<unsigned char>& data) override;
		void writePixels(const std::vector<unsigned char>& data) override;
		void bind(unsigned int texUnit) override;
		void unbind(unsigned int texUnit) override;
		void* getHandle() override;
	};
}

#endif