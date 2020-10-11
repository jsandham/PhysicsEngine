#ifndef __GRAPHICS_H__
#define __GRAPHICS_H__

#include "../glm/glm.hpp"

#include "../core/Cubemap.h"
#include "../core/Font.h"
#include "../core/Material.h"
#include "../core/Mesh.h"
#include "../core/Shader.h"
#include "../core/Texture2D.h"
#include "../core/Texture3D.h"
#include "../core/World.h"

#include "GraphicsQuery.h"
#include "GraphicsState.h"
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

class Graphics
{
  public:
    static void checkError();
    static void checkFrambufferError();
    static GLenum getTextureFormat(TextureFormat format);

    static void beginQuery(GLuint queryId);
    static void endQuery(GLuint queryId, GLuint64 *elapsedTime);

    static void createTargets(CameraTargets *targets, Viewport viewport, glm::vec3 *ssaoSamples, GLuint *queryId0,
                              GLuint *queryId1, bool *created);
    static void destroyTargets(CameraTargets *targets, GLuint *queryId0, GLuint *queryId1, bool *created);
    static void resizeTargets(CameraTargets *targets, Viewport viewport, bool *viewportChanged);
    static void readColorPickingPixel(const CameraTargets *targets, int x, int y, Color32 *color);

    static void createTargets(LightTargets *targets, ShadowMapResolution resolution, bool *created);
    static void destroyTargets(LightTargets *targets, bool *created);
    static void resizeTargets(LightTargets *targets, ShadowMapResolution resolution, bool *resolutionChanged);

    static void create(Texture2D *texture, GLuint *tex, bool *created);
    static void destroy(Texture2D *texture, GLuint *tex, bool *created);
    static void readPixels(Texture2D *texture);
    static void apply(Texture2D *texture);

    static void create(Texture3D *texture, GLuint *tex, bool *created);
    static void destroy(Texture3D *texture, GLuint *tex, bool *created);
    static void readPixels(Texture3D *texture);
    static void apply(Texture3D *texture);

    static void create(Cubemap *cubemap, GLuint *tex, bool *created);
    static void destroy(Cubemap *cubemap, GLuint *tex, bool *created);
    static void readPixels(Cubemap *cubemap);
    static void apply(Cubemap *cubemap);

    static void create(Mesh *mesh, GLuint *vao, GLuint *vbo0, GLuint *vbo1, GLuint *vbo2, bool *created);
    static void destroy(Mesh *mesh, GLuint *vao, GLuint *vbo0, GLuint *vbo1, GLuint *vbo2, bool *created);
    static void apply(Mesh *mesh);

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

    static void render(World *world, RenderObject renderObject, GraphicsQuery *query);
};
} // namespace PhysicsEngine

#endif