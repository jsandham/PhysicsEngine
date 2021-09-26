#ifndef GRAPHICS_H__
#define GRAPHICS_H__

#include <string>

#include "GL/glew.h"

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"

#include "../components/Camera.h"
#include "../components/Light.h"
#include "../core/Shader.h"
#include "../core/Texture.h"
#include "../core/RenderTexture.h"

#include "GraphicsQuery.h"
#include "RenderObject.h"

namespace PhysicsEngine
{
enum API
{
    OpenGL,
    DirectX
};

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

    unsigned int mBuffer;
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
    float mShadowBias;            // 788
    float mShadowRadius;           // 792
    float mShadowStrength;         // 796

    unsigned int mBuffer;
};

class Graphics
{
  public:
    static void checkError(long line, const char *file);
    static void checkFrambufferError(long line, const char *file);
    static GLenum getTextureFormat(TextureFormat format);
    static int getTextureWrapMode(TextureWrapMode wrapMode);
    static int getTextureFilterMode(TextureFilterMode filterMode);
    static void beginQuery(unsigned int queryId);
    static void endQuery(unsigned int queryId, unsigned long long*elapsedTime);
    static void createGlobalCameraUniforms(CameraUniform &uniform);
    static void createGlobalLightUniforms(LightUniform &uniform);
    static void setGlobalCameraUniforms(const CameraUniform &uniform);
    static void setGlobalLightUniforms(const LightUniform &uniform);
    static void createScreenQuad(unsigned int* vao, unsigned int* vbo);
    static void renderScreenQuad(unsigned int vao);
    static void createFramebuffer(int width, int height, unsigned int* fbo, unsigned int* color, unsigned int* depth);
    static void destroyFramebuffer(unsigned int* fbo, unsigned int* color, unsigned int* depth);
    static void bindFramebuffer(unsigned int fbo);
    static void unbindFramebuffer();
    static void clearFrambufferColor(const Color &color);
    static void clearFrambufferColor(float r, float g, float b, float a);
    static void clearFramebufferDepth(float depth);
    static void setViewport(int x, int y, int width, int height);
    static void createTargets(CameraTargets *targets, Viewport viewport, glm::vec3 *ssaoSamples, unsigned int *queryId0,
                              unsigned int *queryId1);
    static void destroyTargets(CameraTargets *targets, unsigned int *queryId0, unsigned int *queryId1);
    static void resizeTargets(CameraTargets *targets, Viewport viewport, bool *viewportChanged);
    static void readColorPickingPixel(const CameraTargets *targets, int x, int y, Color32 *color);
    static void createTargets(LightTargets *targets, ShadowMapResolution resolution);
    static void destroyTargets(LightTargets *targets);
    static void resizeTargets(LightTargets *targets, ShadowMapResolution resolution);
    static void createTexture2D(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                                int height, const std::vector<unsigned char> &data, unsigned int *tex);
    static void destroyTexture2D(unsigned int *tex);
    static void updateTexture2D(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, unsigned int tex);
    static void readPixelsTexture2D(TextureFormat format, int width, int height, int numChannels,
                                    std::vector<unsigned char> &data, unsigned int tex);
    static void writePixelsTexture2D(TextureFormat format, int width, int height,
                                     const std::vector<unsigned char> &data, unsigned int tex);
    static void createTexture3D(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                                int height, int depth, const std::vector<unsigned char> &data, unsigned int*tex);
    static void destroyTexture3D(unsigned int*tex);
    static void updateTexture3D(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, unsigned int tex);
    static void readPixelsTexture3D(TextureFormat format, int width, int height, int depth, int numChannels,
                                    std::vector<unsigned char> &data, unsigned int tex);
    static void writePixelsTexture3D(TextureFormat format, int width, int height, int depth,
                                     const std::vector<unsigned char> &data, unsigned int tex);
    static void createCubemap(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
                              const std::vector<unsigned char> &data, unsigned int*tex);
    static void destroyCubemap(unsigned int*tex);
    static void updateCubemap(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, unsigned int tex);
    static void readPixelsCubemap(TextureFormat format, int width, int numChannels, std::vector<unsigned char> &data,
                                unsigned int tex);
    static void writePixelsCubemap(TextureFormat format, int width, const std::vector<unsigned char> &data, unsigned int tex);



    static void createRenderTextureTargets(RenderTextureTargets* targets, TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width, int height);
    static void destroyRenderTextureTargets(RenderTextureTargets* targets);


    static void createMesh(const std::vector<float> &vertices, const std::vector<float> &normals,
                           const std::vector<float> &texCoords, unsigned int*vao, unsigned int*vbo0, unsigned int*vbo1, unsigned int*vbo2);
    static void destroyMesh(unsigned int*vao, unsigned int*vbo0, unsigned int*vbo1, unsigned int*vbo2);
    static void createSprite(unsigned int*vao);
    static void destroySprite(unsigned int*vao);
    static void preprocess(std::string& vert, std::string& frag, std::string& geom, int64_t variant);
    static bool compile(const std::string &name, const std::string &vert, const std::string &frag, const std::string &geom, unsigned int *program);
    static int findUniformLocation(const char *name, int program);
    static int getUniformCount(int program);
    static int getAttributeCount(int program);
    static std::vector<ShaderUniform> getShaderUniforms(int program);
    static std::vector<ShaderAttribute> getShaderAttributes(int program);
    static void setUniformBlock(const char *blockName, int bindingPoint, int program);
    static void use(int program);
    static void unuse();
    static void destroy(int program);
    static void setBool(int nameLocation, bool value);
    static void setInt(int nameLocation, int value);
    static void setFloat(int nameLocation, float value);
    static void setColor(int nameLocation, const Color &color);
    static void setColor32(int nameLocation, const Color32& color);
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
    static Color32 getColor32(int nameLocation, int program);
    static glm::vec2 getVec2(int nameLocation, int program);
    static glm::vec3 getVec3(int nameLocation, int program);
    static glm::vec4 getVec4(int nameLocation, int program);
    static glm::mat2 getMat2(int nameLocation, int program);
    static glm::mat3 getMat3(int nameLocation, int program);
    static glm::mat4 getMat4(int nameLocation, int program);
    static int getTexture2D(int nameLocation, int texUnit, int program);
    static void applyMaterial(const std::vector<ShaderUniform> &uniforms, const std::vector<int> &textures,
                              int shaderProgram);
    static void render(int start, int count, int vao, bool wireframe = false);
    static void render(const RenderObject &renderObject, GraphicsQuery &query);
};
} // namespace PhysicsEngine

#endif
























//static void (*checkError)(long, const char*);
//static void (*checkFrambufferError)(long, const char*);
//static GLenum(*getTextureFormat)(TextureFormat);
//static GLint(*getTextureWrapMode)(TextureWrapMode);
//static GLint(*getTextureFilterMode)(TextureFilterMode);
//static void (*beginQuery)(GLuint);
//static void (*endQuery)(GLuint, GLuint64*);
//static void (*createGlobalCameraUniforms)(CameraUniform&);
//static void (*createGlobalLightUniforms)(LightUniform&);
//static void (*setGlobalCameraUniforms)(const CameraUniform&);
//static void (*setGlobalLightUniforms)(const LightUniform&);
//static void (*createScreenQuad)(GLuint*, GLuint*);
//static void (*renderScreenQuad)(GLuint);
//static void (*createFramebuffer)(int, int, GLuint*, GLuint*, GLuint*);
//static void (*destroyFramebuffer)(GLuint*, GLuint*, GLuint*);
//static void (*bindFramebuffer)(GLuint);
//static void (*unbindFramebuffer)();
//static void (*clearFrambufferColor)(const Color&);
//static void (*clearFrambufferColor)(float, float, float, float);
//static void (*clearFramebufferDepth)(float);
//static void (*setViewport)(int, int, int, int);
//static void (*createTargets)(CameraTargets*, Viewport, glm::vec3*, GLuint*, GLuint*);
//static void (*destroyTargets)(CameraTargets*, GLuint*, GLuint*);
//static void (*resizeTargets)(CameraTargets*, Viewport, bool*);
//static void (*readColorPickingPixel)(const CameraTargets*, int, int, Color32*);
//static void (*createTargets)(LightTargets*, ShadowMapResolution);
//static void (*destroyTargets)(LightTargets*);
//static void (*resizeTargets)(LightTargets*, ShadowMapResolution);
//static void (*createTexture2D)(TextureFormat, TextureWrapMode, TextureFilterMode, int, int, const std::vector<unsigned char>&, GLuint*);
//static void (*destroyTexture2D)(GLuint*);
//static void (*updateTexture2D)(TextureWrapMode, TextureFilterMode, int, GLuint);
//static void (*readPixelsTexture2D)(TextureFormat, int, int, int, std::vector<unsigned char>&, GLuint);
//static void (*writePixelsTexture2D)(TextureFormat, int, int, const std::vector<unsigned char>&, GLuint);
//static void (*createTexture3D)(TextureFormat, TextureWrapMode, TextureFilterMode, int, int, int, const std::vector<unsigned char>&, GLuint*);
//static void (*destroyTexture3D)(GLuint*);
//static void (*updateTexture3D)(TextureWrapMode, TextureFilterMode, int, GLuint);
//static void (*readPixelsTexture3D)(TextureFormat, int, int, int, int,
//    std::vector<unsigned char>& data, GLuint);
//static void (*writePixelsTexture3D)(TextureFormat format, int width, int height, int depth,
//    const std::vector<unsigned char>& data, GLuint tex);
//static void (*createCubemap)(TextureFormat format, TextureWrapMode wrapMode, TextureFilterMode filterMode, int width,
//    const std::vector<unsigned char>& data, GLuint* tex);
//static void (*destroyCubemap)(GLuint* tex);
//static void (*updateCubemap)(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel, GLuint tex);
//static void (*readPixelsCubemap)(TextureFormat format, int width, int numChannels, std::vector<unsigned char>& data,
//    GLuint tex);
//static void (*writePixelsCubemap)(TextureFormat format, int width, const std::vector<unsigned char>& data, GLuint tex);
//static void (*createMesh)(const std::vector<float>& vertices, const std::vector<float>& normals,
//    const std::vector<float>& texCoords, GLuint* vao, GLuint* vbo0, GLuint* vbo1, GLuint* vbo2);
//static void (*destroyMesh)(GLuint* vao, GLuint* vbo0, GLuint* vbo1, GLuint* vbo2);
//static void (*createSprite)(GLuint* vao);
//static void (*destroySprite)(GLuint* vao);
//static bool (*compile)(const std::string& vert, const std::string& frag, const std::string& geom, GLuint* program);
//static int (*findUniformLocation)(const char* name, int program);
//static int (*getUniformCount)(int program);
//static int (*getAttributeCount)(int program);
//static std::vector<Uniform>(*getUniforms)(int program);
//static std::vector<Attribute>(*getAttributes)(int program);
//static void (*setUniformBlock)(const char* blockName, int bindingPoint, int program);
//static void (*use)(int program);
//static void (*unuse)();
//static void (*destroy)(int program);
//static void (*setBool)(int nameLocation, bool value);
//static void (*setInt)(int nameLocation, int value);
//static void (*setFloat)(int nameLocation, float value);
//static void (*setColor)(int nameLocation, const Color& color);
//static void (*setVec2)(int nameLocation, const glm::vec2& vec);
//static void (*setVec3)(int nameLocation, const glm::vec3& vec);
//static void (*setVec4)(int nameLocation, const glm::vec4& vec);
//static void (*setMat2)(int nameLocation, const glm::mat2& mat);
//static void (*setMat3)(int nameLocation, const glm::mat3& mat);
//static void (*setMat4)(int nameLocation, const glm::mat4& mat);
//static void (*setTexture2D)(int nameLocation, int texUnit, int tex);
//static bool (*getBool)(int nameLocation, int program);
//static int (*getInt)(int nameLocation, int program);
//static float (*getFloat)(int nameLocation, int program);
//static Color(*getColor)(int nameLocation, int program);
//static glm::vec2(*getVec2)(int nameLocation, int program);
//static glm::vec3(*getVec3)(int nameLocation, int program);
//static glm::vec4(*getVec4)(int nameLocation, int program);
//static glm::mat2(*getMat2)(int nameLocation, int program);
//static glm::mat3(*getMat3)(int nameLocation, int program);
//static glm::mat4(*getMat4)(int nameLocation, int program);
//static int (*getTexture2D)(int nameLocation, int texUnit, int program);
//static void (*applyMaterial)(const std::vector<ShaderUniform>& uniforms, const std::vector<GLint>& textures,
//    int shaderProgram);
//static void (*render)(int start, int count, GLuint vao, bool wireframe = false);
//static void (*render)(const RenderObject& renderObject, GraphicsQuery& query);
// function pointer idea
//void (*checkError)(long, const char*);
//void (*checkFrambufferError)(long, const char*);
//checkError = &checkError_opengl;
//checkFrambufferError = &checkFrambufferError_opengl;
// or
//checkError = &checkError_directx;
//checkFrambufferError = &checkFrambufferError_directx;