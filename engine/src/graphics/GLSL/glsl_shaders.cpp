//***************************************
// THIS IS A GENERATED FILE. DO NOT EDIT.
//***************************************
#include <string>
#include "glsl_shaders.h"
using namespace PhysicsEngine;
std::string PhysicsEngine::getColorFragmentShader()
{
return "#version 430 core\n"
"struct Material\n"
"{\n"
"    uvec4 color;\n"
"};\n"
"uniform Material material;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"    FragColor = vec4(material.color.r / 255.0f, material.color.g / 255.0f,\n"
"                      material.color.b / 255.0f, material.color.a / 255.0f);\n"
"}\n";
}
std::string PhysicsEngine::getColorInstancedFragmentShader()
{
return "#version 430 core\n"
"\n"
"in vec4 Color;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"    FragColor = vec4(Color.r / 255.0f, Color.g / 255.0f,\n"
"                      Color.b / 255.0f, Color.a / 255.0f);\n"
"}\n";
}
std::string PhysicsEngine::getColorInstancedVertexShader()
{
return "#version 430 core\n"
"layout(std140) uniform CameraBlock\n"
"{\n"
"    mat4 projection;\n"
"    mat4 view;\n"
"    mat4 viewProjection;\n"
"    vec3 cameraPos;\n"
"}Camera;\n"
"\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 3) in mat4 model;\n"
"layout (location = 7) in vec4 color;\n"
"\n"
"out vec4 Color;\n"
"\n"
"void main()\n"
"{\n"
"    Color = color;\n"
"    gl_Position = Camera.viewProjection * model * vec4(position, 1.0);\n"
"}\n";
}
std::string PhysicsEngine::getColorVertexShader()
{
return "#version 430 core\n"
"layout(std140) uniform CameraBlock\n"
"{\n"
"    mat4 projection;\n"
"    mat4 view;\n"
"    mat4 viewProjection;\n"
"    vec3 cameraPos;\n"
"}Camera;\n"
"\n"
"layout (location = 0) in vec3 position;\n"
"uniform mat4 model;\n"
"void main()\n"
"{\n"
"    gl_Position = Camera.viewProjection * model * vec4(position, 1.0);\n"
"}\n";
}
std::string PhysicsEngine::getGBufferFragmentShader()
{
return "#version 430 core\n"
"struct Material\n"
"{\n"
"    float shininess;\n"
"    vec3 ambient;\n"
"    vec3 diffuse;\n"
"    vec3 specular;\n"
"    vec3 colour;\n"
"    sampler2D mainTexture;\n"
"    sampler2D normalMap;\n"
"    sampler2D specularMap;\n"
"\n"
"    int sampleMainTexture;\n"
"    int sampleNormalMap;\n"
"    int sampleSpecularMap;\n"
"};\n"
"layout(location = 0) out vec3 gPosition;\n"
"layout(location = 1) out vec3 gNormal;\n"
"layout(location = 2) out vec4 gAlbedoSpec;\n"
"in vec2 TexCoords;\n"
"in vec3 FragPos;\n"
"in vec3 Normal;\n"
"\n"
"uniform Material material;\n"
"void main()\n"
"{\n"
"  // store the fragment position vector in the first gbuffer texture\n"
"  gPosition = FragPos;\n"
"  // also store the per-fragment normals into the gbuffer\n"
"  gNormal = normalize(Normal);\n"
"  // and the diffuse per-fragment color\n"
"  gAlbedoSpec.rgb = texture(material.mainTexture, TexCoords).rgb;\n"
"  // store specular intensity in gAlbedoSpec's alpha component\n"
"  gAlbedoSpec.a = 1.0;//texture(texture_specular1, TexCoords).r;\n"
"}\n";
}
std::string PhysicsEngine::getGBufferVertexShader()
{
return "#version 430 core\n"
"layout(std140) uniform CameraBlock\n"
"{\n"
"    mat4 projection;\n"
"    mat4 view;\n"
"    mat4 viewProjection;\n"
"    vec3 cameraPos;\n"
"}Camera;\n"
"layout(location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec3 aNormal;\n"
"layout(location = 2) in vec2 aTexCoords;\n"
"\n"
"out vec3 FragPos;\n"
"out vec2 TexCoords;\n"
"out vec3 Normal;\n"
"uniform mat4 model;\n"
"void main()\n"
"{\n"
"    vec4 worldPos = model * vec4(aPos, 1.0);\n"
"    FragPos = worldPos.xyz;\n"
"    TexCoords = aTexCoords;\n"
"    mat3 normalMatrix = transpose(inverse(mat3(model)));\n"
"    Normal = normalMatrix * aNormal;\n"
"    gl_Position = Camera.projection * Camera.view * worldPos;\n"
"}\n"
"\n";
}
std::string PhysicsEngine::getGeometryFragmentShader()
{
return "#version 430 core\n"
"layout(location = 0) out vec3 positionTex;\n"
"layout(location = 1) out vec3 normalTex;\n"
"in vec3 FragPos;\n"
"in vec3 Normal;\n"
"void main()\n"
"{\n"
"   // store the fragment position vector in the first gbuffer texture\n"
"   positionTex = FragPos.xyz;\n"
"   // also store the per-fragment normals into the gbuffer\n"
"   normalTex = normalize(Normal);\n"
"}\n";
}
std::string PhysicsEngine::getGeometryVertexShader()
{
return "#version 430 core\n"
"layout(std140) uniform CameraBlock\n"
"{\n"
"    mat4 projection;\n"
"    mat4 view;\n"
"    mat4 viewProjection;\n"
"    vec3 cameraPos;\n"
"}Camera;\n"
"\n"
"in vec3 position;\n"
"in vec3 normal;\n"
"in vec2 texCoord;\n"
"out vec3 FragPos;\n"
"out vec3 Normal;\n"
"uniform mat4 model;\n"
"void main()\n"
"{\n"
"    vec4 viewPos = Camera.view * model * vec4(position, 1.0);\n"
"    FragPos = viewPos.xyz;\n"
"    mat3 normalMatrix = transpose(inverse(mat3(Camera.view * model)));\n"
"    Normal = normalMatrix * normal;\n"
"    gl_Position = Camera.projection * viewPos;\n"
"}\n";
}
std::string PhysicsEngine::getGizmoFragmentShader()
{
return "#version 430 core\n"
"out vec4 FragColor;\n"
"in vec3 Normal;\n"
"in vec3 FragPos;\n"
"uniform vec3 lightPos;\n"
"uniform vec4 color;\n"
"void main()\n"
"{\n"
"  vec3 norm = normalize(Normal);\n"
"  vec3 lightDir = normalize(lightPos - FragPos);\n"
"  float diff = max(abs(dot(norm, lightDir)), 0.1);\n"
"  vec4 diffuse = vec4(diff, diff, diff, 1.0);\n"
"  FragColor = diffuse * color;\n"
"}\n";
}
std::string PhysicsEngine::getGizmoVertexShader()
{
return "#version 430 core\n"
"layout(location = 0) in vec3 position;\n"
"layout(location = 1) in vec3 normal;\n"
"out vec3 FragPos;\n"
"out vec3 Normal;\n"
"uniform mat4 model;\n"
"uniform mat4 view;\n"
"uniform mat4 projection;\n"
"void main()\n"
"{\n"
"    FragPos = vec3(model * vec4(position, 1.0));\n"
"    Normal = mat3(transpose(inverse(model))) * normal;\n"
"    gl_Position = projection * view * vec4(FragPos, 1.0);\n"
"}\n";
}
std::string PhysicsEngine::getGridFragmentShader()
{
return "#version 430 core\n"
"in vec4 Color;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"  float depth = 0.2f * gl_FragCoord.z / gl_FragCoord.w;\n"
"  FragColor = vec4(Color.x, Color.y, Color.z, clamp(1.0f / depth, 0.0f, 0.8f));\n"
"}\n";
}
std::string PhysicsEngine::getGridVertexShader()
{
return "#version 430 core\n"
"layout(std140) uniform CameraBlock\n"
"{\n"
"    mat4 projection;\n"
"    mat4 view;\n"
"    mat4 viewProjection;\n"
"    vec3 cameraPos;\n"
"}Camera;\n"
"\n"
"uniform mat4 mvp;\n"
"uniform vec4 color;\n"
"in vec3 position;\n"
"out vec4 Color;\n"
"void main()\n"
"{\n"
"    gl_Position = mvp * vec4(position, 1.0);\n"
"    Color = color;\n"
"}\n";
}
std::string PhysicsEngine::getLinearDepthFragmentShader()
{
return "#version 430 core\n"
"out vec4 FragColor;\n"
"float near = 0.1;\n"
"float far = 100.0;\n"
"float LinearizeDepth(float depth)\n"
"{\n"
"  float z = depth * 2.0 - 1.0; // back to NDC\n"
"  return (2.0 * near * far) / (far + near - z * (far - near));\n"
"}\n"
"void main()\n"
"{\n"
"  float depth = LinearizeDepth(gl_FragCoord.z) / far;\n"
"  FragColor = vec4(vec3(depth), 1.0);\n"
"}\n";
}
std::string PhysicsEngine::getLinearDepthInstancedFragmentShader()
{
return "#version 430 core\n"
"out vec4 FragColor;\n"
"float near = 0.1;\n"
"float far = 100.0;\n"
"float LinearizeDepth(float depth)\n"
"{\n"
"  float z = depth * 2.0 - 1.0; // back to NDC\n"
"  return (2.0 * near * far) / (far + near - z * (far - near));\n"
"}\n"
"void main()\n"
"{\n"
"  float depth = LinearizeDepth(gl_FragCoord.z) / far;\n"
"  FragColor = vec4(vec3(depth), 1.0);\n"
"}\n";
}
std::string PhysicsEngine::getLinearDepthInstancedVertexShader()
{
return "#version 430 core\n"
"layout(std140) uniform CameraBlock\n"
"{\n"
"    mat4 projection;\n"
"    mat4 view;\n"
"    mat4 viewProjection;\n"
"    vec3 cameraPos;\n"
"}Camera;\n"
"layout(location = 3) in mat4 model;\n"
"\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"    gl_Position = Camera.viewProjection * model * vec4(position, 1.0);\n"
"}\n";
}
std::string PhysicsEngine::getLinearDepthVertexShader()
{
return "#version 430 core\n"
"layout(std140) uniform CameraBlock\n"
"{\n"
"    mat4 projection;\n"
"    mat4 view;\n"
"    mat4 viewProjection;\n"
"    vec3 cameraPos;\n"
"}Camera;\n"
"\n"
"uniform mat4 model;\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"    gl_Position = Camera.viewProjection * model * vec4(position, 1.0);\n"
"}\n";
}
std::string PhysicsEngine::getLineFragmentShader()
{
return "#version 430 core\n"
"in vec4 Color;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"  FragColor = Color;\n"
"}\n";
}
std::string PhysicsEngine::getLineVertexShader()
{
return "#version 430 core\n"
"layout(location = 0) in vec3 position;\n"
"layout(location = 1) in vec4 color;\n"
"uniform mat4 mvp;\n"
"out vec4 Color;\n"
"void main()\n"
"{\n"
"    Color = color;\n"
"    gl_Position = mvp * vec4(position, 1.0);\n"
"}\n";
}
std::string PhysicsEngine::getNormalFragmentShader()
{
return "#version 430 core\n"
"in vec3 Normal;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"  FragColor = vec4(normalize(Normal), 1);\n"
"}\n";
}
std::string PhysicsEngine::getNormalInstancedFragmentShader()
{
return "#version 430 core\n"
"in vec3 Normal;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"  FragColor = vec4(normalize(Normal), 1);\n"
"}\n";
}
std::string PhysicsEngine::getNormalInstancedVertexShader()
{
return "#version 430 core\n"
"layout(std140) uniform CameraBlock\n"
"{\n"
"    mat4 projection;\n"
"    mat4 view;\n"
"    mat4 viewProjection;\n"
"    vec3 cameraPos;\n"
"}Camera;\n"
"layout(location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec3 aNormal;\n"
"layout(location = 3) in mat4 model;\n"
"\n"
"out vec3 Normal;\n"
"void main()\n"
"{\n"
"    vec4 worldPos = model * vec4(aPos, 1.0);\n"
"    mat3 normalMatrix = transpose(inverse(mat3(model)));\n"
"    Normal = normalMatrix * aNormal;\n"
"    gl_Position = Camera.viewProjection * worldPos;\n"
"}\n"
"\n";
}
std::string PhysicsEngine::getNormalVertexShader()
{
return "#version 430 core\n"
"layout(std140) uniform CameraBlock\n"
"{\n"
"    mat4 projection;\n"
"    mat4 view;\n"
"    mat4 viewProjection;\n"
"    vec3 cameraPos;\n"
"}Camera;\n"
"layout(location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec3 aNormal;\n"
"\n"
"out vec3 Normal;\n"
"uniform mat4 model;\n"
"void main()\n"
"{\n"
"    vec4 worldPos = model * vec4(aPos, 1.0);\n"
"    mat3 normalMatrix = transpose(inverse(mat3(model)));\n"
"    Normal = normalMatrix * aNormal;\n"
"    gl_Position = Camera.viewProjection * worldPos;\n"
"}\n"
"\n";
}
std::string PhysicsEngine::getPositionFragmentShader()
{
return "#version 430 core\n"
"in vec3 FragPos;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"  FragColor = vec4(FragPos, 1);\n"
"}\n";
}
std::string PhysicsEngine::getPositionInstancedFragmentShader()
{
return "#version 430 core\n"
"in vec3 FragPos;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"  FragColor = vec4(FragPos, 1);\n"
"}\n";
}
std::string PhysicsEngine::getPositionInstancedVertexShader()
{
return "#version 430 core\n"
"layout(std140) uniform CameraBlock\n"
"{\n"
"    mat4 projection;\n"
"    mat4 view;\n"
"    mat4 viewProjection;\n"
"    vec3 cameraPos;\n"
"}Camera;\n"
"layout(location = 0) in vec3 aPos;\n"
"layout(location = 3) in mat4 model;\n"
"\n"
"out vec3 FragPos;\n"
"void main()\n"
"{\n"
"    vec4 worldPos = model * vec4(aPos, 1.0);\n"
"    FragPos = worldPos.xyz;\n"
"    gl_Position = Camera.viewProjection * worldPos;\n"
"}\n";
}
std::string PhysicsEngine::getPositionVertexShader()
{
return "#version 430 core\n"
"layout(std140) uniform CameraBlock\n"
"{\n"
"    mat4 projection;\n"
"    mat4 view;\n"
"    mat4 viewProjection;\n"
"    vec3 cameraPos;\n"
"}Camera;\n"
"layout(location = 0) in vec3 aPos;\n"
"\n"
"out vec3 FragPos;\n"
"uniform mat4 model;\n"
"void main()\n"
"{\n"
"    vec4 worldPos = model * vec4(aPos, 1.0);\n"
"    FragPos = worldPos.xyz;\n"
"    gl_Position = Camera.viewProjection * worldPos;\n"
"}\n";
}
std::string PhysicsEngine::getScreenQuadFragmentShader()
{
return "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec2 TexCoords;\n"
"uniform sampler2D screenTexture;\n"
"void main()\n"
"{\n"
"  vec3 col = texture(screenTexture, TexCoords).rgb;\n"
"  FragColor = vec4(col, 1.0);\n"
"}\n";
}
std::string PhysicsEngine::getScreenQuadVertexShader()
{
return "#version 330 core\n"
"layout(location = 0) in vec2 aPos;\n"
"layout(location = 1) in vec2 aTexCoords;\n"
"out vec2 TexCoords;\n"
"void main()\n"
"{\n"
"    TexCoords = aTexCoords;\n"
"    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);\n"
"}\n";
}
std::string PhysicsEngine::getShadowDepthCubemapFragmentShader()
{
return "#version 430 core\n"
"in vec4 FragPos;\n"
"uniform vec3 lightPos;\n"
"uniform float farPlane;\n"
"void main()\n"
"{\n"
"  float lightDistance = length(FragPos.xyz - lightPos);\n"
"  lightDistance = lightDistance / farPlane;\n"
"  gl_FragDepth = lightDistance;\n"
"}\n"
"\n";
}
std::string PhysicsEngine::getShadowDepthCubemapGeometryShader()
{
return "#version 430 core\n"
"layout(triangles) in;\n"
"layout(triangle_strip, max_vertices = 18) out;\n"
"uniform mat4 cubeViewProjMatrices[6];\n"
"out vec4 FragPos;\n"
"void main()\n"
"{\n"
"  for (int i = 0; i < 6; i++)\n"
"  {\n"
"      gl_Layer = i;\n"
"      for (int j = 0; j < 3; j++)\n"
"      {\n"
"          FragPos = gl_in[j].gl_Position;\n"
"          gl_Position = cubeViewProjMatrices[i] * FragPos;\n"
"          EmitVertex();\n"
"      }\n"
"      EndPrimitive();\n"
"  }\n"
"}\n";
}
std::string PhysicsEngine::getShadowDepthCubemapVertexShader()
{
return "#version 430 core\n"
"in vec3 position;\n"
"uniform mat4 model;\n"
"void main()\n"
"{\n"
"    gl_Position = model * vec4(position, 1.0);\n"
"}\n"
"\n";
}
std::string PhysicsEngine::getShadowDepthMapFragmentShader()
{
return "#version 430 core\n"
"void main()\n"
"{\n"
"}\n";
}
std::string PhysicsEngine::getShadowDepthMapVertexShader()
{
return "#version 430 core\n"
"uniform mat4 projection;\n"
"uniform mat4 view;\n"
"uniform mat4 model;\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"    gl_Position = projection * view * model * vec4(position, 1.0);\n"
"}\n";
}
std::string PhysicsEngine::getSpriteFragmentShader()
{
return "#version 430 core\n"
"in vec2 TexCoords;\n"
"out vec4 color;\n"
"uniform sampler2D image;\n"
"uniform vec4 spriteColor;\n"
"void main()\n"
"{\n"
"  color = spriteColor * texture(image, TexCoords);\n"
"}\n";
}
std::string PhysicsEngine::getSpriteVertexShader()
{
return "#version 430 core\n"
"layout(location = 0) in vec4 vertex; // <vec2 position, vec2 texCoords>\n"
"out vec2 TexCoords;\n"
"uniform mat4 model;\n"
"uniform mat4 view;\n"
"uniform mat4 projection;\n"
"void main()\n"
"{\n"
"    TexCoords = vertex.zw;\n"
"    gl_Position = projection * view * model * vec4(vertex.xy, 0.0, 1.0);\n"
"}\n";
}
std::string PhysicsEngine::getSSAOFragmentShader()
{
return "#version 430 core\n"
"out float FragColor;\n"
"in vec2 TexCoord;\n"
"uniform sampler2D positionTex;\n"
"uniform sampler2D normalTex;\n"
"uniform sampler2D noiseTex;\n"
"uniform vec3 samples[64];\n"
"// parameters (you'd probably want to use them as uniforms to more easily tweak the effect)\n"
"int kernelSize = 64;\n"
"float radius = 0.5;\n"
"float bias = 0.025;\n"
"// tile noise texture over screen based on screen dimensions divided by noise size\n"
"const vec2 noiseScale = vec2(1024.0 / 4.0, 1024.0 / 4.0);\n"
"uniform mat4 projection;\n"
"void main()\n"
"{\n"
"   // get input for SSAO algorithm\n"
"   vec3 fragPos = texture(positionTex, TexCoord).xyz;\n"
"   vec3 normal = normalize(texture(normalTex, TexCoord).rgb);\n"
"   vec3 randomVec = normalize(texture(noiseTex, TexCoord * noiseScale).xyz);\n"
"   // create TBN change-of-basis matrix: from tangent-space to view-space\n"
"   vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));\n"
"   vec3 bitangent = cross(normal, tangent);\n"
"   mat3 TBN = mat3(tangent, bitangent, normal);\n"
"   // iterate over the sample kernel and calculate occlusion factor\n"
"   float occlusion = 0.0f;\n"
"   for (int i = 0; i < kernelSize; ++i)\n"
"   {\n"
"       // get sample position\n"
"       vec3 sampleq = TBN * samples[i]; // from tangent to view-space\n"
"       sampleq = fragPos + sampleq * radius;\n"
"       // project sample position (to sample texture) (to get position on screen/texture)\n"
"       vec4 offset = vec4(sampleq, 1.0);\n"
"       offset = projection * offset; // from view to clip-space\n"
"       offset.xyz /= offset.w;       // perspective divide\n"
"       offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0\n"
"       // get sample depth\n"
"       float sampleDepth = texture(positionTex, offset.xy).z; // get depth value of kernel sample\n"
"       // range check & accumulate\n"
"       float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));\n"
"       occlusion += (sampleDepth >= sampleq.z + bias ? 1.0 : 0.0) * rangeCheck;\n"
"   }\n"
"   occlusion = 1.0 - (occlusion / kernelSize);\n"
"   FragColor = occlusion;\n"
"}\n";
}
std::string PhysicsEngine::getSSAOVertexShader()
{
return "#version 430 core\n"
"in vec3 position;\n"
"in vec2 texCoord;\n"
"out vec2 TexCoord;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(position, 1.0);\n"
"   TexCoord = texCoord;\n"
"}\n";
}
std::string PhysicsEngine::getStandardFragmentShader()
{
return "#version 430 core\n"
"layout(std140) uniform LightBlock\n"
"{\n"
"    mat4 lightProjection[5]; // 0    64   128  192  256\n"
"    mat4 lightView[5]; // 320  384  448  512  576\n"
"    vec3 position; // 640\n"
"    vec3 direction; // 656\n"
"    vec3 color; // 672\n"
"    float cascadeEnds[5]; // 688  704  720  736  752\n"
"    float intensity; // 768\n"
"    float spotAngle; // 772\n"
"    float innerSpotAngle; // 776\n"
"    float shadowNearPlane; // 780\n"
"    float shadowFarPlane; // 784\n"
"    float shadowBias; // 788\n"
"    float shadowRadius; // 792\n"
"    float shadowStrength; // 796\n"
"}Light;\n"
"struct Material\n"
"{\n"
"    float shininess;\n"
"    vec3 ambient;\n"
"    vec3 diffuse;\n"
"    vec3 specular;\n"
"    vec3 colour;\n"
"    sampler2D mainTexture;\n"
"    sampler2D normalMap;\n"
"    sampler2D specularMap;\n"
"\n"
"    int sampleMainTexture;\n"
"    int sampleNormalMap;\n"
"    int sampleSpecularMap;\n"
"};\n"
"\n"
"uniform Material material;\n"
"uniform sampler2D shadowMap[5];\n"
"in vec3 FragPos;\n"
"in vec3 CameraPos;\n"
"in vec3 Normal;\n"
"in vec2 TexCoord;\n"
"in float ClipSpaceZ;\n"
"in vec4 FragPosLightSpace[5];\n"
"vec2 poissonDisk[4] = vec2[](vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725), vec2(-0.094184101, -0.92938870), vec2(0.34495938, 0.29387760));\n"
"out vec4 FragColor;\n"
"vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir);\n"
"vec3 CalcSpotLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir);\n"
"vec3 CalcPointLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir);\n"
"float CalcShadow(int index, vec4 fragPosLightSpace);\n"
"void main(void)\n"
"{\n"
"    vec3 viewDir = normalize(CameraPos - FragPos);\n"
"    vec4 albedo = vec4(material.colour, 1.0);\n"
"    if(material.sampleMainTexture == 1)\n"
"    {\n"
"        albedo = texture(material.mainTexture, TexCoord);\n"
"    }\n"
"#if defined (DIRECTIONALLIGHT)\n"
"    FragColor = vec4(CalcDirLight(material, Normal, viewDir), 1.0f) * albedo;\n"
"#elif defined(SPOTLIGHT)\n"
"    FragColor = vec4(CalcSpotLight(material, Normal, FragPos, viewDir), 1.0f) * albedo;\n"
"#elif defined(POINTLIGHT)\n"
"    FragColor = vec4(CalcPointLight(material, Normal, FragPos, viewDir), 1.0f) * albedo;\n"
"#else\n"
"    FragColor = vec4(0.5, 0.5, 0.5, 1.0) * albedo;\n"
"#endif\n"
"}\n"
"vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir)\n"
"{\n"
"    vec3 norm = normalize(normal);\n"
"    vec3 lightDir = normalize(-Light.direction);\n"
"    vec3 reflectDir = reflect(-lightDir, norm);\n"
"    float diffuseStrength = max(dot(norm, lightDir), 0.0);\n"
"    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);\n"
"    float shadow = 0.0f;\n"
"#if defined (SOFTSHADOWS) || defined(HARDSHADOWS)\n"
"    for (int i = 0; i < 5; i++)\n"
"    {\n"
"        if(ClipSpaceZ <= Light.cascadeEnds[i])\n"
"        {\n"
"            shadow = Light.shadowStrength * CalcShadow(i, FragPosLightSpace[i]);\n"
"            break;\n"
"        }\n"
"    }\n"
"#endif\n"
"    vec3 ambient = material.ambient;\n"
"    vec3 diffuse = (1.0f - shadow) * material.diffuse * diffuseStrength;\n"
"    vec3 specular = (1.0f - shadow) * material.specular * specularStrength;\n"
"    diffuse = diffuse * Light.intensity * Light.color;\n"
"    specular = specular * Light.intensity * Light.color;\n"
"    vec3 finalColor = (ambient + diffuse + specular);\n"
"#if defined (SHOWCASCADES)\n"
"    if(ClipSpaceZ <= Light.cascadeEnds[0])\n"
"    {\n"
"        finalColor = finalColor * vec3(1.0f, 0.0f, 0.0f);\n"
"    }\n"
"    else if (ClipSpaceZ <= Light.cascadeEnds[1])\n"
"    {\n"
"        finalColor = finalColor * vec3(0.0f, 1.0f, 0.0f);\n"
"    }\n"
"    else if (ClipSpaceZ <= Light.cascadeEnds[2])\n"
"    {\n"
"        finalColor = finalColor * vec3(0.0f, 0.0f, 1.0f);\n"
"    }\n"
"    else if (ClipSpaceZ <= Light.cascadeEnds[3])\n"
"    {\n"
"        finalColor = finalColor * vec3(0.0f, 1.0f, 1.0f);\n"
"    }\n"
"    else if (ClipSpaceZ <= Light.cascadeEnds[4])\n"
"    {\n"
"        finalColor = finalColor * vec3(0.6f, 0.0f, 0.6f);\n"
"    }\n"
"    else\n"
"    {\n"
"        finalColor = vec3(0.5, 0.5, 0.5);\n"
"    }\n"
"#endif\n"
"return finalColor;\n"
"}\n"
"vec3 CalcSpotLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir)\n"
"{\n"
"    vec3 lightDir = normalize(Light.position - fragPos);\n"
"    vec3 reflectDir = reflect(-lightDir, normal);\n"
"    float diffuseStrength = max(dot(normal, lightDir), 0.0);\n"
"    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0f), material.shininess);\n"
"    float theta = dot(lightDir, normalize(-Light.direction));\n"
"    float epsilon = Light.innerSpotAngle - Light.spotAngle;\n"
"    float intensity = clamp((theta - Light.spotAngle) / epsilon, 0.0f, 1.0f);\n"
"    float shadow = 0;\n"
"#if defined (SOFTSHADOWS) || defined(HARDSHADOWS)\n"
"    shadow = Light.shadowStrength * CalcShadow(0, FragPosLightSpace[0]);\n"
"#endif\n"
"    float distance = length(Light.position - fragPos);\n"
"    float attenuation = 1.0f; // / (1.0f + 0.0f * distance + 0.01f * distance * distance);\n"
"    vec3 ambient = material.ambient;\n"
"    vec3 diffuse = (1.0f - shadow) * material.diffuse * diffuseStrength;\n"
"    vec3 specular = (1.0f - shadow) * material.specular * specularStrength;\n"
"    ambient *= attenuation;\n"
"    diffuse *= attenuation * intensity * Light.intensity * Light.color;\n"
"    specular *= attenuation * intensity * Light.intensity * Light.color;\n"
"    return vec3(ambient + diffuse + specular);\n"
"}\n"
"vec3 CalcPointLight(Material material, vec3 normal, vec3 fragPos, vec3 viewDir)\n"
"{\n"
"    vec3 lightDir = normalize(Light.position - fragPos);\n"
"    vec3 reflectDir = reflect(-lightDir, normal);\n"
"    float ambientStrength = 1.0f;\n"
"    float diffuseStrength = max(dot(normal, lightDir), 0.0);\n"
"    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0f), material.shininess);\n"
"    float distance = length(Light.position - fragPos);\n"
"    float attenuation = 1.0f; // 1.0f / (Light.constant + Light.linear * distance + Light.quadratic * distance * distance);\n"
"    vec3 ambient = material.ambient * ambientStrength;\n"
"    vec3 diffuse = material.diffuse * diffuseStrength;\n"
"    vec3 specular = material.specular * specularStrength;\n"
"    // vec3 specular = material.specular * vec3(texture(material.specularMap, TexCoord)) * specularStrength;\n"
"    ambient *= attenuation;\n"
"    diffuse *= attenuation * Light.intensity * Light.color;\n"
"    specular *= attenuation * Light.intensity * Light.color;\n"
"    return vec3(ambient + diffuse + specular);\n"
"}\n"
"float CalcShadow(int index, vec4 fragPosLightSpace)\n"
"{\n"
"    // only actually needed when using perspective projection for the light\n"
"    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;\n"
"    // projCoord is in [-1,1] range. Convert it ot [0,1] range.\n"
"    projCoords = projCoords * 0.5 + 0.5;\n"
"    float closestDepth = texture(shadowMap[index], projCoords.xy).r;\n"
"    // get depth of current fragment from light's perspective\n"
"    float currentDepth = projCoords.z; // - 0.005;\n"
"    // check whether current frag pos is in shadow\n"
"    // float shadow = closestDepth < currentDepth ? 1.0 : 0.0;\n"
"    // float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);\n"
"    float shadow = currentDepth - Light.shadowBias > closestDepth ? 1.0 : 0.0;\n"
"    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.\n"
"    if(projCoords.z > 1.0)\n"
"        shadow = 0.0;\n"
"    return shadow;\n"
"};\n";
}
std::string PhysicsEngine::getStandardVertexShader()
{
return "#version 430 core\n"
"layout(std140) uniform CameraBlock\n"
"{\n"
"    mat4 projection;\n"
"    mat4 view;\n"
"    mat4 viewProjection;\n"
"    vec3 cameraPos;\n"
"}Camera;\n"
"layout(std140) uniform LightBlock\n"
"{\n"
"    mat4 lightProjection[5]; // 0    64   128  192  256\n"
"    mat4 lightView[5]; // 320  384  448  512  576\n"
"    vec3 position; // 640\n"
"    vec3 direction; // 656\n"
"    vec3 color; // 672\n"
"    float cascadeEnds[5]; // 688  704  720  736  752\n"
"    float intensity; // 768\n"
"    float spotAngle; // 772\n"
"    float innerSpotAngle; // 776\n"
"    float shadowNearPlane; // 780\n"
"    float shadowFarPlane; // 784\n"
"    float shadowBias; // 788\n"
"    float shadowRadius; // 792\n"
"    float shadowStrength; // 796\n"
"}Light;\n"
"\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 1) in vec3 normal;\n"
"layout (location = 2) in vec2 texCoord;\n"
"#if defined (INSTANCING)\n"
"layout (location = 3) in mat4 model;\n"
"#else\n"
"uniform mat4 model;\n"
"#endif\n"
"\n"
"out vec3 FragPos;\n"
"out vec3 CameraPos;\n"
"out vec3 Normal;\n"
"out vec2 TexCoord;\n"
"out float ClipSpaceZ;\n"
"out vec4 FragPosLightSpace[5];\n"
"void main()\n"
"{\n"
"    CameraPos = Camera.cameraPos;\n"
"    FragPos = vec3(model * vec4(position, 1.0));\n"
"    Normal = mat3(transpose(inverse(model))) * normal;\n"
"    TexCoord = texCoord;\n"
"    gl_Position = Camera.viewProjection * vec4(FragPos, 1.0);\n"
"    ClipSpaceZ = gl_Position.z;\n"
"    for (int i = 0; i < 5; i++)\n"
"    {\n"
"        FragPosLightSpace[i] = Light.lightProjection[i] * Light.lightView[i] * vec4(FragPos, 1.0f);\n"
"    }\n"
"};\n";
}
