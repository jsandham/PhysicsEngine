#include "../../include/core/InternalShaders.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;


const std::string InternalShaders::gizmoVertexShader = "layout(location = 0) in vec3 position;\n"
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
"}";

const std::string InternalShaders::gizmoFragmentShader ="out vec4 FragColor;\n"
                                                        "in vec3 Normal;\n"
                                                        "in vec3 FragPos;\n"
                                                        "uniform vec3 lightPos;\n"
                                                        "uniform vec4 color;\n"
                                                        "void main()\n"
                                                        "{\n"
                                                        "    vec3 norm = normalize(Normal);\n"
                                                        "    vec3 lightDir = normalize(lightPos - FragPos);\n"
                                                        "    float diff = max(dot(norm, lightDir), 0.1);\n"
                                                        "    vec4 diffuse = vec4(diff, diff, diff, 1.0);\n"
                                                        "    FragColor = diffuse * color;\n"
                                                        "}";

const std::string InternalShaders::colorVertexShader =
    "#define DIRECTIONALLIGHT\n"
    "#define HARDSHADOWS\n"
    "#define SOFTSHADOWS\n"
    "layout (std140) uniform CameraBlock\n"
    "{\n"
    "	mat4 projection;\n"
    "	mat4 view;\n"
    "	vec3 cameraPos;\n"
    "}Camera;\n"
    "uniform mat4 model;\n"
    "in vec3 position;\n"
    "void main()\n"
    "{\n"
    "	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);\n"
    "}";

const std::string InternalShaders::colorFragmentShader = "uniform vec4 color;\n"
                                                         "out vec4 FragColor;\n"
                                                         "void main()\n"
                                                         "{\n"
                                                         "	FragColor = color;\n"
                                                         "}";

const std::string InternalShaders::screenQuadVertexShader = "in vec3 position;\n"
                                                            "in vec2 texCoord;\n"
                                                            "out vec2 TexCoord;\n"
                                                            "void main()\n"
                                                            "{\n"
                                                            "	gl_Position = vec4(position, 1.0);\n"
                                                            "   TexCoord = texCoord;\n"
                                                            "}";

const std::string InternalShaders::screenQuadFragmentShader = "uniform sampler2D texture0;\n"
                                                              "in vec2 TexCoord;\n"
                                                              "out vec4 FragColor;\n"
                                                              "void main()\n"
                                                              "{\n"
                                                              "    FragColor = texture(texture0, TexCoord);\n"
                                                              "}";

const std::string InternalShaders::normalMapVertexShader =
    "layout (std140) uniform CameraBlock\n"
    "{\n"
    "	mat4 projection;\n"
    "	mat4 view;\n"
    "	vec3 cameraPos;\n"
    "}Camera;\n"
    "uniform mat4 model;\n"
    "in vec3 position;\n"
    "in vec3 normal;\n"
    "out vec3 Normal;\n"
    "void main()\n"
    "{\n"
    "	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);\n"
    "   Normal = normal;\n"
    "}";

const std::string InternalShaders::normalMapFragmentShader = "in vec3 Normal;\n"
                                                             "out vec4 FragColor;\n"
                                                             "void main()\n"
                                                             "{\n"
                                                             "	FragColor = vec4(Normal.xyz, 1.0f);\n"
                                                             "}";

const std::string InternalShaders::depthMapVertexShader =
    "layout (std140) uniform CameraBlock\n"
    "{\n"
    "	mat4 projection;\n"
    "	mat4 view;\n"
    "	vec3 cameraPos;\n"
    "}Camera;\n"
    "uniform mat4 model;\n"
    "in vec3 position;\n"
    "void main()\n"
    "{\n"
    "	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);\n"
    "}";

const std::string InternalShaders::depthMapFragmentShader = "void main()\n"
                                                            "{\n"
                                                            "}";

const std::string InternalShaders::shadowDepthMapVertexShader =
    "uniform mat4 projection;\n"
    "uniform mat4 view;\n"
    "uniform mat4 model;\n"
    "in vec3 position;\n"
    "void main()\n"
    "{\n"
    "	gl_Position = projection * view * model * vec4(position, 1.0);\n"
    "}";

const std::string InternalShaders::shadowDepthMapFragmentShader = "void main()\n"
                                                                  "{\n"
                                                                  "}";

const std::string InternalShaders::shadowDepthCubemapVertexShader = "in vec3 position;\n"
                                                                    "uniform mat4 model;\n"
                                                                    "void main()\n"
                                                                    "{\n"
                                                                    "	gl_Position = model * vec4(position, 1.0);\n"
                                                                    "}";

const std::string InternalShaders::shadowDepthCubemapGeometryShader =
    "layout (triangles) in;\n"
    "layout (triangle_strip, max_vertices=18) out;\n"
    "uniform mat4 cubeViewProjMatrices[6];\n"
    "out vec4 FragPos;\n"
    "void main()\n"
    "{\n"
    "	for(int i = 0; i < 6; i++){\n"
    "		gl_Layer = i;\n"
    "		for(int j = 0; j < 3; j++){\n"
    "			FragPos = gl_in[j].gl_Position;\n"
    "			gl_Position = cubeViewProjMatrices[i] * FragPos;\n"
    "			EmitVertex();\n"
    "		}\n"
    "		EndPrimitive();\n"
    "	}\n"
    "}";

const std::string InternalShaders::shadowDepthCubemapFragmentShader =
    "in vec4 FragPos;\n"
    "uniform vec3 lightPos;\n"
    "uniform float farPlane;\n"
    "void main()\n"
    "{\n"
    "	float lightDistance = length(FragPos.xyz - lightPos);\n"
    "   lightDistance = lightDistance / farPlane;\n"
    "   gl_FragDepth = 1.0f;\n"
    "}";

const std::string InternalShaders::overdrawVertexShader =
    "layout (std140) uniform CameraBlock\n"
    "{\n"
    "	mat4 projection;\n"
    "	mat4 view;\n"
    "	vec3 cameraPos;\n"
    "}Camera;\n"
    "uniform mat4 model;\n"
    "in vec3 position;\n"
    "void main()\n"
    "{\n"
    "	gl_Position = Camera.projection * Camera.view * model * vec4(position, 1.0);\n"
    "}";

const std::string InternalShaders::overdrawFragmentShader = "out vec4 FragColor;\n"
                                                            "void main()\n"
                                                            "{\n"
                                                            "	FragColor = vec4(1.0, 0.0, 0.0, 0.1);\n"
                                                            "}";

const std::string InternalShaders::fontVertexShader = "layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>\n"
                                                      "out vec2 TexCoords;\n"
                                                      "uniform mat4 projection;\n"
                                                      "void main()\n"
                                                      "{\n"
                                                      "    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);\n"
                                                      "    TexCoords = vertex.zw;\n"
                                                      "}";

const std::string InternalShaders::fontFragmentShader =
    "in vec2 TexCoords;\n"
    "out vec4 color;\n"
    "uniform sampler2D text;\n"
    "uniform vec3 textColor;\n"
    "void main()\n"
    "{\n"
    "    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);\n"
    "    color = vec4(textColor, 1.0) * sampled;\n"
    "}";

const std::string InternalShaders::gbufferVertexShader =
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec3 aNormal;\n"
    "layout (location = 2) in vec2 aTexCoords;\n"

    "layout (std140) uniform CameraBlock\n"
    "{\n"
    "	mat4 projection;\n"
    "	mat4 view;\n"
    "	vec3 cameraPos;\n"
    "}Camera;\n"

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
    "}\n";

const std::string InternalShaders::gbufferFragmentShader =
    "layout (location = 0) out vec3 gPosition;\n"
    "layout (location = 1) out vec3 gNormal;\n"
    "layout (location = 2) out vec4 gAlbedoSpec;\n"

    "in vec2 TexCoords;\n"
    "in vec3 FragPos;\n"
    "in vec3 Normal;\n"

    "uniform sampler2D texture_diffuse1;\n"
    "uniform sampler2D texture_specular1;\n"

    "void main()\n"
    "{\n"
    "    // store the fragment position vector in the first gbuffer texture\n"
    "    gPosition = FragPos;\n"
    "    // also store the per-fragment normals into the gbuffer\n"
    "    gNormal = normalize(Normal);\n"
    "    // and the diffuse per-fragment color\n"
    "    gAlbedoSpec.rgb = texture(texture_diffuse1, TexCoords).rgb;\n"
    "    // store specular intensity in gAlbedoSpec's alpha component\n"
    "    gAlbedoSpec.a = texture(texture_specular1, TexCoords).r;\n"
    "}\n";

const std::string InternalShaders::positionAndNormalsVertexShader =
    "layout (std140) uniform CameraBlock\n"
    "{\n"
    "	mat4 projection;\n"
    "	mat4 view;\n"
    "	vec3 cameraPos;\n"
    "}Camera;\n"

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

const std::string InternalShaders::positionAndNormalsFragmentShader =
    "layout (location = 0) out vec3 positionTex;\n"
    "layout (location = 1) out vec3 normalTex;\n"

    "in vec3 FragPos;\n"
    "in vec3 Normal;\n"

    "void main()\n"
    "{\n"
    "    // store the fragment position vector in the first gbuffer texture\n"
    "    positionTex = FragPos.xyz;\n"
    "    // also store the per-fragment normals into the gbuffer\n"
    "    normalTex = normalize(Normal);\n"
    "}\n";

const std::string InternalShaders::ssaoVertexShader = "in vec3 position;\n"
                                                      "in vec2 texCoord;\n"

                                                      "out vec2 TexCoord;\n"

                                                      "void main()\n"
                                                      "{\n"
                                                      "	gl_Position = vec4(position, 1.0);\n"
                                                      "   TexCoord = texCoord;\n"
                                                      "}\n";

const std::string InternalShaders::ssaoFragmentShader =
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
    "	// get input for SSAO algorithm\n"
    "	vec3 fragPos = texture(positionTex, TexCoord).xyz;\n"
    "	vec3 normal = normalize(texture(normalTex, TexCoord).rgb);\n"
    "	vec3 randomVec = normalize(texture(noiseTex, TexCoord * noiseScale).xyz);\n"
    "	// create TBN change-of-basis matrix: from tangent-space to view-space\n"
    "	vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));\n"
    "	vec3 bitangent = cross(normal, tangent);\n"
    "	mat3 TBN = mat3(tangent, bitangent, normal);\n"
    "	// iterate over the sample kernel and calculate occlusion factor\n"
    "	float occlusion = 0.0f;\n"
    "	for (int i = 0; i < kernelSize; ++i)\n"
    "	{\n"
    "		// get sample position\n"
    "		vec3 sampleq = TBN * samples[i]; // from tangent to view-space\n"
    "		sampleq = fragPos + sampleq * radius;\n"
    "		// project sample position (to sample texture) (to get position on screen/texture)\n"
    "		vec4 offset = vec4(sampleq, 1.0);\n"
    "		offset = projection * offset; // from view to clip-space\n"
    "		offset.xyz /= offset.w; // perspective divide\n"
    "		offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0\n"
    "		// get sample depth\n"
    "		float sampleDepth = texture(positionTex, offset.xy).z; // get depth value of kernel sample\n"
    "		// range check & accumulate\n"
    "		float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));\n"
    "		occlusion += (sampleDepth >= sampleq.z + bias ? 1.0 : 0.0) * rangeCheck;\n"
    "	}\n"
    "	occlusion = 1.0 - (occlusion / kernelSize);\n"
    "	FragColor = occlusion;\n"
    "}\n";

const std::string InternalShaders::simpleLitVertexShader = "uniform mat4 model;\n"
                                                           "uniform mat4 view;\n"
                                                           "uniform mat4 projection;\n"
                                                           "uniform vec3 cameraPos;\n"

                                                           "in vec3 position;\n"
                                                           "in vec3 normal;\n"
                                                           "in vec2 texCoord;\n"

                                                           "out vec3 FragPos;\n"
                                                           "out vec3 CameraPos;\n"
                                                           "out vec3 Normal;\n"
                                                           "out vec2 TexCoord;\n"

                                                           "void main()\n"
                                                           "{\n"
                                                           "	CameraPos = cameraPos;\n"
                                                           "	FragPos = vec3(model * vec4(position, 1.0));\n"
                                                           "	Normal = mat3(transpose(inverse(model))) * normal;\n"
                                                           "	TexCoord = texCoord;\n"

                                                           "	gl_Position = projection * view * vec4(FragPos, 1.0);\n"
                                                           "}\n";

const std::string InternalShaders::simpleLitFragmentShader = "in vec3 FragPos;\n"
                                                             "in vec3 CameraPos;\n"
                                                             "in vec3 Normal;\n"
                                                             "in vec2 TexCoord;\n"

                                                             "out vec4 FragColor;\n"

                                                             "void main(void)\n"
                                                             "{\n"
                                                             "	FragColor = vec4(1.0, 0.5, 0.5, 1.0);\n"
                                                             "}\n";

// std::string InternalShaders::simpleLitFragmentShader =
//"struct Material\n"
//"{\n"
//"	float shininess;\n"
//"	vec3 ambient;\n"
//"	vec3 diffuse;\n"
//"	vec3 specular;\n"
//
//"	sampler2D mainTexture;\n"
//"	sampler2D normalMap;\n"
//"	sampler2D specularMap;\n"
//"};\n"
//
//"uniform Material material;\n"
//
//"uniform vec3 direction;\n"
//"uniform vec3 ambient;\n"
//"uniform vec3 diffuse;\n"
//"uniform vec3 specular;\n"
//
//"in vec3 FragPos;\n"
//"in vec3 CameraPos;\n"
//"in vec3 Normal;\n"
//"in vec2 TexCoord;\n"
//
//"out vec4 FragColor;\n"
//
//"vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir);\n"
//
//"void main(void)\n"
//"{\n"
//"	vec3 viewDir = normalize(CameraPos - FragPos);\n"
//
//"	FragColor = vec4(CalcDirLight(material, Normal, viewDir), 1.0f) * texture(material.mainTexture, TexCoord);\n"
//"	//FragColor = vec4(0.5, 0.5, 0.5, 1.0);\n"
//"}\n"
//
//"vec3 CalcDirLight(Material material, vec3 normal, vec3 viewDir)\n"
//"{\n"
//"	vec3 norm = normalize(normal);\n"
//"	vec3 lightDir = normalize(direction);\n"
//
//"	vec3 reflectDir = reflect(-lightDir, norm);\n"
//
//"	float ambientStrength = 1.0f;\n"
//"	float diffuseStrength = max(dot(norm, lightDir), 0.0);\n"
//"	float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);\n"
//
//"	vec3 fambient = ambient * material.ambient * ambientStrength;\n"
//"	vec3 fdiffuse = diffuse * material.diffuse * diffuseStrength;\n"
//"	vec3 fspecular = specular * material.specular * vec3(texture(material.specularMap, TexCoord)) * specularStrength;\n"
//
//"	return (fambient + fdiffuse + fspecular);\n"
//"}\n";

const std::string InternalShaders::simpleLitDeferredVertexShader = "layout(location = 0) in vec3 aPos;\n"
                                                                   "layout(location = 1) in vec2 aTexCoords;\n"

                                                                   "out vec2 TexCoords;\n"

                                                                   "void main()\n"
                                                                   "{\n"
                                                                   "	TexCoords = aTexCoords;\n"
                                                                   "	gl_Position = vec4(aPos, 1.0);\n"
                                                                   "}\n";

const std::string InternalShaders::simpleLitDeferredFragmentShader =
    "out vec4 FragColor;\n"

    "in vec2 TexCoords;\n"

    "uniform sampler2D gPosition;\n"
    "uniform sampler2D gNormal;\n"
    "uniform sampler2D gAlbedoSpec;\n"

    "struct Light {\n"
    "	vec3 Position;\n"
    "	vec3 Color;\n"
    "};\n"
    "const int NR_LIGHTS = 32;\n"
    "uniform Light lights[NR_LIGHTS];\n"
    "uniform vec3 viewPos;\n"

    "void main()\n"
    "{\n"
    "	// retrieve data from G-buffer\n"
    "	vec3 FragPos = texture(gPosition, TexCoords).rgb;\n"
    "	vec3 Normal = texture(gNormal, TexCoords).rgb;\n"
    "	vec3 Albedo = texture(gAlbedoSpec, TexCoords).rgb;\n"
    "	float Specular = texture(gAlbedoSpec, TexCoords).a;\n"

    "	// then calculate lighting as usual\n"
    "	vec3 lighting = Albedo * 0.1; // hard-coded ambient component\n"
    "	vec3 viewDir = normalize(viewPos - FragPos);\n"
    "	for (int i = 0; i < NR_LIGHTS; ++i)\n"
    "	{\n"
    "		// diffuse\n"
    "		vec3 lightDir = normalize(lights[i].Position - FragPos);\n"
    "		vec3 diffuse = max(dot(Normal, lightDir), 0.0) * Albedo * lights[i].Color;\n"
    "		lighting += diffuse;\n"
    "	}\n"

    "	FragColor = vec4(lighting, 1.0);\n"
    "}\n";

const Guid InternalShaders::fontShaderId("9b37ce25-cc6c-497c-bed9-7c5dbd13b61f");
const Guid InternalShaders::gizmoShaderId("2e8d5044-ae07-4d62-b7f1-48bd13d05625");
const Guid InternalShaders::colorShaderId("cfb7774e-0f7d-4990-b0ff-45034483ecea");
const Guid InternalShaders::positionAndNormalShaderId("dd530a6e-96ba-4a79-a6f5-5a634cd449b9");
const Guid InternalShaders::ssaoShaderId("dba46e51-a544-4fac-a9a1-e9d3d91244b0");
const Guid InternalShaders::screenQuadShaderId("7b42abd4-2053-47c2-bc1b-71db2771a3f4");
const Guid InternalShaders::normalMapShaderId("20ec40c9-6ced-4b47-a68b-acba2234809d");
const Guid InternalShaders::depthMapShaderId("725b606b-db20-4dc3-a550-08836e86cf0d");
const Guid InternalShaders::shadowDepthMapShaderId("bdf4bd00-f3ae-4e57-b558-e8569ab05423");
const Guid InternalShaders::shadowDepthCubemapShaderId("a6bd6d1a-a977-45f6-88fb-a3d40c696453");
const Guid InternalShaders::gbufferShaderId("2b794f1c-97b4-4d90-a1a8-e2e391ff154c");
const Guid InternalShaders::simpleLitShaderId("77cc0f14-157a-4364-b156-2543db31b717");
const Guid InternalShaders::simpleLitDeferredShaderId("a0561704-c34b-42ba-b792-c1b940df329d");
const Guid InternalShaders::overdrawShaderId("da3a582e-35e2-412e-9060-1e8cfe183b5a");

Guid InternalShaders::loadInternalShader(World *world, const Guid shaderId, const std::string vertex,
                                         const std::string fragment, const std::string geometry)
{
    // Create temp shader to compute serialized data vector
    Shader temp;
    temp.load(vertex, fragment, geometry);

    std::vector<char> data = temp.serialize(shaderId);

    Shader *shader = world->createAsset<Shader>(data);
    if (shader != NULL)
    {
        return shader->getId();
    }
    else
    {
        Log::error("Could not load internal shader\n");
        return Guid::INVALID;
    }
}

Guid InternalShaders::loadFontShader(World *world)
{
    return loadInternalShader(world, InternalShaders::fontShaderId, InternalShaders::fontVertexShader,
                              InternalShaders::fontFragmentShader, "");
}

Guid InternalShaders::loadGizmoShader(World* world)
{
    return loadInternalShader(world, InternalShaders::gizmoShaderId, InternalShaders::gizmoVertexShader,
        InternalShaders::gizmoFragmentShader, "");
}

Guid InternalShaders::loadColorShader(World *world)
{
    return loadInternalShader(world, InternalShaders::colorShaderId, InternalShaders::colorVertexShader,
                              InternalShaders::colorFragmentShader, "");
}

Guid InternalShaders::loadPositionAndNormalsShader(World *world)
{
    return loadInternalShader(world, InternalShaders::positionAndNormalShaderId,
                              InternalShaders::positionAndNormalsVertexShader,
                              InternalShaders::positionAndNormalsFragmentShader, "");
}

Guid InternalShaders::loadSsaoShader(World *world)
{
    return loadInternalShader(world, InternalShaders::ssaoShaderId, InternalShaders::ssaoVertexShader,
                              InternalShaders::ssaoFragmentShader, "");
}

Guid InternalShaders::loadScreenQuadShader(World *world)
{
    return loadInternalShader(world, InternalShaders::screenQuadShaderId, InternalShaders::screenQuadVertexShader,
                              InternalShaders::screenQuadFragmentShader, "");
}

Guid InternalShaders::loadNormalMapShader(World *world)
{
    return loadInternalShader(world, InternalShaders::normalMapShaderId, InternalShaders::normalMapVertexShader,
                              InternalShaders::normalMapFragmentShader, "");
}

Guid InternalShaders::loadDepthMapShader(World *world)
{
    return loadInternalShader(world, InternalShaders::depthMapShaderId, InternalShaders::depthMapVertexShader,
                              InternalShaders::depthMapFragmentShader, "");
}

Guid InternalShaders::loadShadowDepthMapShader(World *world)
{
    return loadInternalShader(world, InternalShaders::shadowDepthMapShaderId,
                              InternalShaders::shadowDepthMapVertexShader,
                              InternalShaders::shadowDepthMapFragmentShader, "");
}

Guid InternalShaders::loadShadowDepthCubemapShader(World *world)
{
    return loadInternalShader(
        world, InternalShaders::shadowDepthCubemapShaderId, InternalShaders::shadowDepthCubemapVertexShader,
        InternalShaders::shadowDepthCubemapFragmentShader, InternalShaders::shadowDepthCubemapGeometryShader);
}

Guid InternalShaders::loadGBufferShader(World *world)
{
    return loadInternalShader(world, InternalShaders::gbufferShaderId, InternalShaders::gbufferVertexShader,
                              InternalShaders::gbufferFragmentShader, "");
}

Guid InternalShaders::loadSimpleLitShader(World *world)
{
    return loadInternalShader(world, InternalShaders::simpleLitShaderId, InternalShaders::simpleLitVertexShader,
                              InternalShaders::simpleLitFragmentShader, "");
}

Guid InternalShaders::loadSimpleLitDeferredShader(World *world)
{
    return loadInternalShader(world, InternalShaders::simpleLitDeferredShaderId,
                              InternalShaders::simpleLitDeferredVertexShader,
                              InternalShaders::simpleLitDeferredFragmentShader, "");
}

Guid InternalShaders::loadOverdrawShader(World *world)
{
    return loadInternalShader(world, InternalShaders::overdrawShaderId, InternalShaders::overdrawVertexShader,
                              InternalShaders::overdrawFragmentShader, "");
}