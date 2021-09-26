#ifndef RENDER_TEXTURE_H__
#define RENDER_TEXTURE_H__

#include "Texture.h"

namespace PhysicsEngine
{
    struct RenderTextureTargets
    {
        unsigned int mMainFBO;
        unsigned int mColorTex;
        unsigned int mDepthTex;
    };

    class RenderTexture : public Texture
    {
    private:
        RenderTextureTargets mTargets;
        int mWidth;
        int mHeight;

    public:
        RenderTexture(World* world);
        RenderTexture(World* world, Guid id);
        RenderTexture(World* world, int width, int height);
        RenderTexture(World* world, int width, int height, TextureFormat format);
        ~RenderTexture();

        virtual void serialize(YAML::Node& out) const override;
        virtual void deserialize(const YAML::Node& in) override;

        virtual int getType() const override;
        virtual std::string getObjectName() const override;

        void writeToPNG(const std::string& filepath) const;
        void writeToJPG(const std::string& filepath) const;
        void writeToBMP(const std::string& filepath) const;

        int getWidth() const;
        int getHeight() const;

        void create() override;
        void destroy() override;
        void update() override;
        void readPixels() override;
        void writePixels() override;

        unsigned int getNativeGraphicsMainFBO() const;
        unsigned int getNativeGraphicsColorTex() const;
        unsigned int getNativeGraphicsDepthTex() const;
	};

    template <> struct AssetType<RenderTexture>
    {
        static constexpr int type = PhysicsEngine::RENDER_TEXTURE_TYPE;
    };

    template <> struct IsAssetInternal<RenderTexture>
    {
        static constexpr bool value = true;
    };
}

#endif