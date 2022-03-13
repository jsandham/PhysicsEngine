#ifndef GRAPHICS_H__
#define GRAPHICS_H__

#include <string>

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
enum class API
{
    OpenGL,
    DirectX
};

enum class Capability
{
    Depth_Testing,
    Blending,
    BackfaceCulling,
    LineSmoothing
};

enum class BlendingFactor
{
    ZERO,
    ONE,
    SRC_ALPHA,
    ONE_MINUS_SRC_ALPHA
};

struct CameraUniform
{
    glm::mat4 mProjection;     // 0
    glm::mat4 mView;           // 64
    glm::mat4 mViewProjection; // 128
    glm::vec3 mCameraPos;      // 192

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
    float mShadowBias;             // 788
    float mShadowRadius;           // 792
    float mShadowStrength;         // 796

    unsigned int mBuffer;
};

struct ForwardRendererState
{
    // internal graphics camera state
    CameraUniform mCameraState;

    // internal graphics light state
    LightUniform mLightState;

    // directional light cascade shadow map data
    float mCascadeEnds[6];
    glm::mat4 mCascadeOrthoProj[5];
    glm::mat4 mCascadeLightView[5];

    int mDepthShaderProgram;
    int mDepthShaderModelLoc;
    int mDepthShaderViewLoc;
    int mDepthShaderProjectionLoc;

    // spotlight shadow map data
    glm::mat4 mShadowViewMatrix;
    glm::mat4 mShadowProjMatrix;

    // pointlight cubemap shadow map data
    glm::mat4 mCubeViewProjMatrices[6];

    int mDepthCubemapShaderProgram;
    int mDepthCubemapShaderLightPosLoc;
    int mDepthCubemapShaderFarPlaneLoc;
    int mDepthCubemapShaderModelLoc;
    int mDepthCubemapShaderCubeViewProjMatricesLoc0;
    int mDepthCubemapShaderCubeViewProjMatricesLoc1;
    int mDepthCubemapShaderCubeViewProjMatricesLoc2;
    int mDepthCubemapShaderCubeViewProjMatricesLoc3;
    int mDepthCubemapShaderCubeViewProjMatricesLoc4;
    int mDepthCubemapShaderCubeViewProjMatricesLoc5;

    int mGeometryShaderProgram;
    int mGeometryShaderModelLoc;

    // color picking
    int mColorShaderProgram;
    int mColorInstancedShaderProgram;
    int mColorShaderModelLoc;
    int mColorShaderColorLoc;

    // ssao
    int mSsaoShaderProgram;
    int mSsaoShaderProjectionLoc;
    int mSsaoShaderPositionTexLoc;
    int mSsaoShaderNormalTexLoc;
    int mSsaoShaderNoiseTexLoc;
    int mSsaoShaderSamplesLoc[64];

    // sprite
    int mSpriteShaderProgram;
    int mSpriteModelLoc;
    int mSpriteViewLoc;
    int mSpriteProjectionLoc;
    int mSpriteColorLoc;
    int mSpriteImageLoc;

    // quad
    unsigned int mQuadVAO;
    unsigned int mQuadVBO;
    int mQuadShaderProgram;
    int mQuadShaderTexLoc;
};

struct DeferredRendererState
{
    // internal graphics camera state
    CameraUniform mCameraState;

    int mGBufferShaderProgram;
    int mGBufferShaderModelLoc;
    int mGBufferShaderDiffuseTexLoc;
    int mGBufferShaderSpecTexLoc;

    int mSimpleLitDeferredShaderProgram;
    int mSimpleLitDeferredShaderViewPosLoc;
    int mSimpleLitDeferredShaderLightLocs[32];

    // color picking
    int mColorShaderProgram;
    int mColorInstancedShaderProgram;
    int mColorShaderModelLoc;
    int mColorShaderColorLoc;

    // quad
    unsigned int mQuadVAO;
    unsigned int mQuadVBO;
    int mQuadShaderProgram;
    int mQuadShaderTexLoc;
};

struct DebugRendererState
{
    // internal graphics camera state
    CameraUniform mCameraState;

    // normals
    int mNormalsShaderProgram;
    int mNormalsInstancedShaderProgram;
    int mNormalsShaderModelLoc;

    // position
    int mPositionShaderProgram;
    int mPositionInstancedShaderProgram;
    int mPositionShaderModelLoc;

    // linear depth
    int mLinearDepthShaderProgram;
    int mLinearDepthInstancedShaderProgram;
    int mLinearDepthShaderModelLoc;

    // color picking
    int mColorShaderProgram;
    int mColorInstancedShaderProgram;
    int mColorShaderModelLoc;
    int mColorShaderColorLoc;

    // quad
    unsigned int mQuadVAO;
    unsigned int mQuadVBO;
    int mQuadShaderProgram;
    int mQuadShaderTexLoc;
};

struct GizmoRendererState
{
    int mLineShaderProgram;
    int mLineShaderMVPLoc;

    int mGizmoShaderProgram;
    int mGizmoShaderModelLoc;
    int mGizmoShaderViewLoc;
    int mGizmoShaderProjLoc;
    int mGizmoShaderColorLoc;
    int mGizmoShaderLightPosLoc;

    int mGridShaderProgram;
    int mGridShaderMVPLoc;
    int mGridShaderColorLoc;

    unsigned int mFrustumVAO;
    unsigned int mFrustumVBO[2];
    std::vector<float> mFrustumVertices;
    std::vector<float> mFrustumNormals;

    unsigned int mGridVAO;
    unsigned int mGridVBO;
    std::vector<glm::vec3> mGridVertices;
    glm::vec3 mGridOrigin;
    Color mGridColor;
};

class Graphics
{
  public:
    static int INSTANCE_BATCH_SIZE;

    static void checkError(long line, const char *file);
    static void checkFrambufferError(long line, const char *file);
    static void turnOn(Capability capability);
    static void turnOff(Capability capability);
    static void setBlending(BlendingFactor source, BlendingFactor dest);
    static void beginQuery(unsigned int queryId);
    static void endQuery(unsigned int queryId, unsigned long long*elapsedTime);
    static void createGlobalCameraUniforms(CameraUniform &uniform);
    static void createGlobalLightUniforms(LightUniform &uniform);
    static void setGlobalCameraUniforms(const CameraUniform &uniform);
    static void setGlobalLightUniforms(const LightUniform &uniform);
    static void createScreenQuad(unsigned int* vao, unsigned int* vbo);
    static void renderScreenQuad(unsigned int vao);
    static void createFramebuffer(int width, int height, unsigned int *fbo, unsigned int *color);
    static void createFramebuffer(int width, int height, unsigned int* fbo, unsigned int* color, unsigned int* depth);
    static void destroyFramebuffer(unsigned int* fbo, unsigned int* color, unsigned int* depth);
    static void bindFramebuffer(unsigned int fbo);
    static void unbindFramebuffer();
    static void clearFrambufferColor(const Color &color);
    static void clearFrambufferColor(float r, float g, float b, float a);
    static void clearFramebufferDepth(float depth);
    static void bindVertexArray(unsigned int vao);
    static void unbindVertexArray();
    static void setViewport(int x, int y, int width, int height);
    static void createTargets(CameraTargets *targets, Viewport viewport, glm::vec3 *ssaoSamples, unsigned int *queryId0,
                              unsigned int *queryId1);
    static void destroyTargets(CameraTargets *targets, unsigned int *queryId0, unsigned int *queryId1);
    static void resizeTargets(CameraTargets *targets, Viewport viewport, bool *viewportChanged);
    static void readColorAtPixel(const unsigned int *fbo, int x, int y, Color32 *color);
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
    static void createTerrainChunk(const std::vector<float> &vertices, const std::vector<float> &normals,
                           const std::vector<float> &texCoords, int vertexCount, 
                           unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1, unsigned int *vbo2);
    static void destroyTerrainChunk(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1, unsigned int *vbo2);
    static void updateTerrainChunk(const std::vector<float> &vertices, const std::vector<float> &normals,
                                   unsigned int vbo0, unsigned int vbo1);
    static void updateTerrainChunk(const std::vector<float> &vertices, const std::vector<float> &normals,
                                   const std::vector<float> &texCoords, unsigned int vbo0, unsigned int vbo1, 
                                   unsigned int vbo2);
    static void createMesh(const std::vector<float> &vertices, const std::vector<float> &normals,
                           const std::vector<float> &texCoords, unsigned int*vao, unsigned int*vbo0, unsigned int*vbo1, unsigned int*vbo2, unsigned int*model_vbo, unsigned int*color_vbo);
    static void destroyMesh(unsigned int*vao, unsigned int*vbo0, unsigned int*vbo1, unsigned int*vbo2, unsigned int*model_vbo, unsigned int*color_vbo);
    static void updateInstanceBuffer(unsigned int vbo, const glm::mat4* models, size_t instanceCount);
    static void updateInstanceColorBuffer(unsigned int vbo, const glm::vec4 *colors, size_t instanceCount);
    static void createSprite(unsigned int*vao);
    static void destroySprite(unsigned int*vao);
    static void createFrustum(const std::vector<float> &vertices, const std::vector<float> &normals, unsigned int *vao,
                              unsigned int *vbo0, unsigned int *vbo1);
    static void destroyFrustum(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1);
    static void updateFrustum(const std::vector<float> &vertices, const std::vector<float> &normals, unsigned int vbo0,
                              unsigned int vbo1);
    static void updateFrustum(const std::vector<float> &vertices, unsigned int vbo0);
    static void createGrid(const std::vector<glm::vec3> &vertices, unsigned int *vao, unsigned int *vbo0);
    static void destroyGrid(unsigned int *vao, unsigned int *vbo0);
    static void createLine(const std::vector<float> &vertices, const std::vector<float> &colors, unsigned int *vao,
                           unsigned int *vbo0, unsigned int *vbo1);
    static void destroyLine(unsigned int *vao, unsigned int *vbo0, unsigned int *vbo1);
    
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
    static void setTexture2Ds(int nameLocation, int *texUnits, int count, int *texs);
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
    static void applyMaterial(const std::vector<ShaderUniform> &uniforms, int shaderProgram);
    static void renderLines(int start, int count, int vao);
    static void renderLinesWithCurrentlyBoundVAO(int start, int count);
    static void renderWithCurrentlyBoundVAO(int start, int count);
    static void render(int start, int count, int vao, bool wireframe = false);
    static void render(int start, int count, int vao, GraphicsQuery &query, bool wireframe = false);
    static void renderInstanced(int start, int count, int instanceCount, int vao, GraphicsQuery &query);
    static void render(const RenderObject &renderObject, GraphicsQuery &query);
    static void renderInstanced(const RenderObject &renderObject, GraphicsQuery &query);

    //enum class InternalShader
    // {
    //     SSAO,
    //     ShadowDepthMap,
    //     ShadowDepthCubeMap
    // }
    //static void compileShader(InternalShader shader, RenderState& state);
    // or
    // static void compileShader(InternalShader shader, ForwardRendererState& state);
    // static void compileShader(InternalShader shader, DeferredRendererState& state);
    // static void compileShader(InternalShader shader, DebugRendererState& state);
    // static void compileShader(InternalShader shader, GizmoRendererState& state);

    /*template <class T> 
    static void compileShader(InternalShader shader, T& state);
    template <> compileShader<ForwardRendererState>(InternalShader shader, ForwardRendererState &state);
    template <> compileShader<DeferredRendererState>(InternalShader shader, DeferredRendererState &state);
    template <> compileShader<DebugRendererState>(InternalShader shader, DebugRendererState &state);
    template <> compileShader<GizmoRendererState>(InternalShader shader, GizmoRendererState &state);*/

    static void compileSSAOShader(ForwardRendererState &state);
    static void compileShadowDepthMapShader(ForwardRendererState &state);
    static void compileShadowDepthCubemapShader(ForwardRendererState &state);
    static void compileColorShader(ForwardRendererState &state);
    static void compileColorInstancedShader(ForwardRendererState &state);
    static void compileScreenQuadShader(ForwardRendererState &state);
    static void compileSpriteShader(ForwardRendererState &state);
    static void compileGBufferShader(DeferredRendererState &state);
    static void compileScreenQuadShader(DeferredRendererState &state);
    static void compileColorShader(DeferredRendererState &state);
    static void compileColorInstancedShader(DeferredRendererState &state);
    static void compileNormalShader(DebugRendererState &state);
    static void compileNormalInstancedShader(DebugRendererState &state);
    static void compilePositionShader(DebugRendererState &state);
    static void compilePositionInstancedShader(DebugRendererState &state);
    static void compileLinearDepthShader(DebugRendererState &state);
    static void compileLinearDepthInstancedShader(DebugRendererState &state);
    static void compileColorShader(DebugRendererState &state);
    static void compileColorInstancedShader(DebugRendererState &state);
    static void compileScreenQuadShader(DebugRendererState &state);
    static void compileLineShader(GizmoRendererState &state);
    static void compileGizmoShader(GizmoRendererState &state);
    static void compileGridShader(GizmoRendererState &state);

    static std::string getGeometryVertexShader();
    static std::string getGeometryFragmentShader();
    static std::string getSSAOVertexShader();
    static std::string getSSAOFragmentShader();
    static std::string getShadowDepthMapVertexShader();
    static std::string getShadowDepthMapFragmentShader();
    static std::string getShadowDepthCubemapVertexShader();
    static std::string getShadowDepthCubemapFragmentShader();
    static std::string getShadowDepthCubemapGeometryShader();
    static std::string getColorVertexShader();
    static std::string getColorFragmentShader();
    static std::string getColorInstancedVertexShader();
    static std::string getColorInstancedFragmentShader();
    static std::string getScreenQuadVertexShader();
    static std::string getScreenQuadFragmentShader();
    static std::string getSpriteVertexShader();
    static std::string getSpriteFragmentShader();
    static std::string getGBufferVertexShader();
    static std::string getGBufferFragmentShader();
    static std::string getNormalVertexShader();
    static std::string getNormalFragmentShader();
    static std::string getNormalInstancedVertexShader();
    static std::string getNormalInstancedFragmentShader();
    static std::string getPositionVertexShader();
    static std::string getPositionFragmentShader();
    static std::string getPositionInstancedVertexShader();
    static std::string getPositionInstancedFragmentShader();
    static std::string getLinearDepthVertexShader();
    static std::string getLinearDepthFragmentShader();
    static std::string getLinearDepthInstancedVertexShader();
    static std::string getLinearDepthInstancedFragmentShader();
    static std::string getLineVertexShader();
    static std::string getLineFragmentShader();
    static std::string getGizmoVertexShader();
    static std::string getGizmoFragmentShader();
    static std::string getGridVertexShader();
    static std::string getGridFragmentShader();
    static std::string getStandardVertexShader();
    static std::string getStandardFragmentShader();
    
};
} // namespace PhysicsEngine

#endif
