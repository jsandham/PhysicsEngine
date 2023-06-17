//#ifndef DIRECTX_RENDER_TEXTURE_HANDLE_H__
//#define DIRECTX_RENDER_TEXTURE_HANDLE_H__
//
//#include "../../RenderTextureHandle.h"
//
//#include <windows.h>
//#include <d3d11.h>
//
//namespace PhysicsEngine
//{
//	class DirectXRenderTextureHandle : public RenderTextureHandle
//	{
//	private:
//		D3D11_TEXTURE2D_DESC mTextureDesc;
//		ID3D11Texture2D* mTexture;
//
//		D3D11_SHADER_RESOURCE_VIEW_DESC mShaderResourceViewDesc;
//		ID3D11ShaderResourceView* mShaderResourceView;
//
//		D3D11_SAMPLER_DESC mSamplerDesc;
//		ID3D11SamplerState* mSamplerState;
//
//	public:
//        DirectXRenderTextureHandle(int width, int height, TextureFormat format, TextureWrapMode wrapMode,
//                                   TextureFilterMode filterMode);
//		~DirectXRenderTextureHandle();
//
//		void* getTexture() override;
//		void* getIMGUITexture() override;
//	};
//}
//
//#endif