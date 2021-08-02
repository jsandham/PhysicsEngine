#ifndef GRAPHICS_H__
#define GRAPHICS_H__

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

#include "../components/Camera.h"
#include "../components/Light.h"
#include "../core/Shader.h"
#include "../core/Texture.h"

#include "GraphicsQuery.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
struct Uniform
{
    GLsizei nameLength;
    GLint size;
    GLenum type;
    GLchar name[32];
};

struct Attribute
{
    GLsizei nameLength;
    GLint size;
    GLenum type;
    GLchar name[32];
};

struct CameraUniform
{
    glm::mat4 mProjection; // 0
    glm::mat4 mView;       // 64
    glm::vec3 mCameraPos;  // 128

    GLuint mBuffer;
};

struct LightUniform
{
    glm::mat4 mLightProjection[5]; // 0    64   128  192  256
    glm::mat4 mLightView[5];       // 320  384  448  512  576
    glm::vec3 mPosition;           // 640
    glm::vec3 mDirection;          // 656
    glm::vec4 mColor;              // 672
    float mCascadeEnds[5];         // 688  704  720  736  752
    float mIntensity;              // 768
    float mSpotAngle;              // 772
    float mInnerSpotAngle;         // 776
    float mShadowNearPlane;        // 780
    float mShadowFarPlane;         // 784
    float mShadowAngle;            // 788
    float mShadowRadius;           // 792
    float mShadowStrength;         // 796

    GLuint mBuffer;
};

class Graphics
{
  public:
    static void checkError(long line, const char *file);
    static void checkFrambufferError(long line, const char *file);
    static GLenum getTextureFormat(TextureFormat format);
    static GLint getTextureWrapMode(TextureWrapMode wrapMode);
    static GLint getTextureFilterMode(TextureFilterMode filterMode);

    static void beginQuery(GLuint queryId);
    static void endQuery(GLuint queryId, GLuint64 *elapsedTime);

    static void createGlobalCameraUniforms(CameraUniform &uniform);
    static void createGlobalLightUniforms(LightUniform &uniform);
    static void setGlobalCameraUniforms(const CameraUniform &uniform);
    static void setGlobalLightUniforms(const LightUniform &uniform);

    static void createScreenQuad(GLuint *vao, GLuint *vbo);
    static void renderScreenQuad(GLuint vao);

    static void createFramebuffer(int width, int height, GLuint *fbo, GLuint *color, GLuint *depth);
    static void destroyFramebuffer(GLuint *fbo, GLuint *color, GLuint *depth);
    static void bindFramebuffer(GLuint fbo);
    static void unbindFramebuffer();
    static void clearFrambufferColor(const Color &color);
    static void clearFrambufferColor(float r, float g, float b, float a);
    static void clearFramebufferDepth(float depth);

    static void setViewport(int x, int y, int width, int height);

    static void createTargets(CameraTargets *targets, Viewport viewport, glm::vec3 *ssaoSamples, GLuint *queryId0,
                              GLuint *queryId1);
    static void destroyTargets(CameraTargets *targets, GLuint *queryId0, GLuint *queryId1);
    static void resizeTargets(CameraTargets *targets, Viewport viewport, bool *viewportChanged);
    static void readColorPickingPixel(const CameraTargets *targets, int x, int y, Color32 *color);

    static void createTargets(LightTargets *targets, ShadowMapResolution resolution);
    static void destroyTargets(LightTargets *targets);
    static void resizeTargets(LightTargets *targets, ShadowMapResolution resolution);

    static void createTexture2D(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                                int height, const std::vector<unsigned char> &data, GLuint *tex);
    static void destroyTexture2D(GLuint *tex);
    static void updateTexture2D(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, GLuint tex);
    static void readPixelsTexture2D(TextureFormat format, int width, int height, int numChannels,
                                    std::vector<unsigned char> &data, GLuint tex);
    static void writePixelsTexture2D(TextureFormat format, int width, int height,
                                     const std::vector<unsigned char> &data, GLuint tex);

    static void createTexture3D(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                                int height, int depth, const std::vector<unsigned char> &data, GLuint *tex);
    static void destroyTexture3D(GLuint *tex);
    static void updateTexture3D(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, GLuint tex);
    static void readPixelsTexture3D(TextureFormat format, int width, int height, int depth, int numChannels,
                                    std::vector<unsigned char> &data, GLuint tex);
    static void writePixelsTexture3D(TextureFormat format, int width, int height, int depth,
                                     const std::vector<unsigned char> &data, GLuint tex);

    static void createCubemap(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                              const std::vector<unsigned char> &data, GLuint *tex);
    static void destroyCubemap(GLuint *tex);
    static void updateCubemap(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, GLuint tex);
    static void readPixelsCubemap(TextureFormat format, int width, int numChannels, std::vector<unsigned char> &data,
                                  GLuint tex);
    static void writePixelsCubemap(TextureFormat format, int width, const std::vector<unsigned char> &data, GLuint tex);

    static void createMesh(const std::vector<float> &vertices, const std::vector<float> &normals,
                           const std::vector<float> &texCoords, GLuint *vao, GLuint *vbo0, GLuint *vbo1, GLuint *vbo2);
    static void destroyMesh(GLuint *vao, GLuint *vbo0, GLuint *vbo1, GLuint *vbo2);

    static void createSprite(GLuint *vao);
    static void destroySprite(GLuint *vao);

    static bool compile(const std::string &vert, const std::string &frag, const std::string &geom, GLuint *program);
    static int findUniformLocation(const char *name, int program);
    static int getUniformCount(int program);
    static int getAttributeCount(int program);
    static std::vector<Uniform> getUniforms(int program);
    static std::vector<Attribute> getAttributes(int program);
    static void setUniformBlock(const char *blockName, int bindingPoint, int program);
    static void use(int program);
    static void unuse();
    static void destroy(int program);
    static void setBool(int nameLocation, bool value);
    static void setInt(int nameLocation, int value);
    static void setFloat(int nameLocation, float value);
    static void setColor(int nameLocation, const Color &color);
    static void setVec2(int nameLocation, const glm::vec2 &vec);
    static void setVec3(int nameLocation, const glm::vec3 &vec);
    static void setVec4(int nameLocation, const glm::vec4 &vec);
    static void setMat2(int nameLocation, const glm::mat2 &mat);
    static void setMat3(int nameLocation, const glm::mat3 &mat);
    static void setMat4(int nameLocation, const glm::mat4 &mat);
    static void setTexture2D(int nameLocation, int texUnit, int tex);
    static bool getBool(int nameLocation, int program);
    static int getInt(int nameLocation, int program);
    static float getFloat(int nameLocation, int program);
    static Color getColor(int nameLocation, int program);
    static glm::vec2 getVec2(int nameLocation, int program);
    static glm::vec3 getVec3(int nameLocation, int program);
    static glm::vec4 getVec4(int nameLocation, int program);
    static glm::mat2 getMat2(int nameLocation, int program);
    static glm::mat3 getMat3(int nameLocation, int program);
    static glm::mat4 getMat4(int nameLocation, int program);
    static int getTexture2D(int nameLocation, int texUnit, int program);

    static void applyMaterial(const std::vector<ShaderUniform> &uniforms, const std::vector<GLint> &textures,
                              int shaderProgram);

    static void render(int start, int count, GLuint vao, bool wireframe = false);
    static void render(const RenderObject &renderObject, GraphicsQuery &query);
};
} // namespace PhysicsEngine

#endif