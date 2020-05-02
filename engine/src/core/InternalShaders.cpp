#include "../../include/core/InternalShaders.h"

using namespace PhysicsEngine;

std::string InternalShaders::lineVertexShader =
"layout (std140) uniform CameraBlock\n"
"{\n"
"	mat4 projection;\n"
"	mat4 view;\n"
"	vec3 cameraPos;\n"
"}Camera;\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = Camera.projection * Camera.view * vec4(position, 1.0);\n"
"}";

std::string InternalShaders::lineFragmentShader =
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
"}";


std::string InternalShaders::colorVertexShader =
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

std::string InternalShaders::colorFragmentShader =
"uniform vec4 color;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = color;\n"
"}";






std::string InternalShaders::graphVertexShader =
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(position, 1.0);\n"
"}";

std::string InternalShaders::graphFragmentShader =
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
"}";

std::string InternalShaders::windowVertexShader =
"in vec3 position;\n"
"in vec2 texCoord;\n"
"out vec2 TexCoord;\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(position, 1.0);\n"
"   TexCoord = texCoord;\n"
"}";

std::string InternalShaders::windowFragmentShader =
"uniform sampler2D texture0;\n"
"in vec2 TexCoord;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"    FragColor = texture(texture0, TexCoord);\n"
"}";

std::string InternalShaders::normalMapVertexShader =
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

std::string InternalShaders::normalMapFragmentShader =
"in vec3 Normal;\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(Normal.xyz, 1.0f);\n"
"}";

std::string InternalShaders::depthMapVertexShader =
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

std::string InternalShaders::depthMapFragmentShader =
"void main()\n"
"{\n"
"}";

std::string InternalShaders::shadowDepthMapVertexShader =
"uniform mat4 projection;\n"
"uniform mat4 view;\n"
"uniform mat4 model;\n"
"in vec3 position;\n"
"void main()\n"
"{\n"
"	gl_Position = projection * view * model * vec4(position, 1.0);\n"
"}";

std::string InternalShaders::shadowDepthMapFragmentShader =
"void main()\n"
"{\n"
"}";






std::string InternalShaders::shadowDepthCubemapVertexShader =
"in vec3 position;\n"
"uniform mat4 model;\n"
"void main()\n"
"{\n"
"	gl_Position = model * vec4(position, 1.0);\n"
"}";

std::string InternalShaders::shadowDepthCubemapGeometryShader =
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

std::string InternalShaders::shadowDepthCubemapFragmentShader =
"in vec4 FragPos;\n"
"uniform vec3 lightPos;\n"
"uniform float farPlane;\n"
"void main()\n"
"{\n"
"	float lightDistance = length(FragPos.xyz - lightPos);\n"
"   lightDistance = lightDistance / farPlane;\n"
"   gl_FragDepth = 1.0f;\n"
"}";







std::string InternalShaders::overdrawVertexShader =
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

std::string InternalShaders::overdrawFragmentShader =
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"	FragColor = vec4(1.0, 0.0, 0.0, 0.1);\n"
"}";

std::string InternalShaders::fontVertexShader =
"layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>\n"
"out vec2 TexCoords;\n"
"uniform mat4 projection;\n"
"void main()\n"
"{\n"
"    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);\n"
"    TexCoords = vertex.zw;\n"
"}";

std::string InternalShaders::fontFragmentShader =
"in vec2 TexCoords;\n"
"out vec4 color;\n"
"uniform sampler2D text;\n"
"uniform vec3 textColor;\n"
"void main()\n"
"{\n"
"    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);\n"
"    color = vec4(textColor, 1.0) * sampled;\n"
"}";

std::string InternalShaders::instanceVertexShader =
"out vec4 FragColor;\n"
"in vec3 fColor;\n"
"void main()\n"
"{\n"
"    FragColor = vec4(fColor, 1.0);\n"
"}";

std::string InternalShaders::instanceFragmentShader =
"layout (location = 0) in vec2 aPos;\n"
"layout (location = 1) in vec3 aColor;\n"
"layout (location = 2) in vec2 aOffset;\n"
"out vec3 fColor;\n"
"void main()\n"
"{\n"
"    gl_Position = vec4(aPos + aOffset, 0.0, 1.0);\n"
"    fColor = aColor;\n"
"}";





std::string InternalShaders::gbufferVertexShader =
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec3 aNormal;\n"
"layout (location = 2) in vec2 aTexCoords;\n"

"out vec3 FragPos;\n"
"out vec2 TexCoords;\n"
"out vec3 Normal;\n"

"uniform mat4 model;\n"
"uniform mat4 view;\n"
"uniform mat4 projection;\n"

"void main()\n"
"{\n"
"    vec4 worldPos = model * vec4(aPos, 1.0);\n"
"    FragPos = worldPos.xyz;\n"
"    TexCoords = aTexCoords;\n"

"    mat3 normalMatrix = transpose(inverse(mat3(model)));\n"
"    Normal = normalMatrix * aNormal;\n"

"    gl_Position = projection * view * worldPos;\n"
"}\n";

std::string InternalShaders::gbufferFragmentShader =
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


std::string InternalShaders::positionAndNormalsVertexShader =
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

std::string InternalShaders::positionAndNormalsFragmentShader =
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


std::string InternalShaders::ssaoVertexShader =
"in vec3 position;\n"
"in vec2 texCoord;\n"

"out vec2 TexCoord;\n"

"void main()\n"
"{\n"
"	gl_Position = vec4(position, 1.0);\n"
"   TexCoord = texCoord;\n"
"}\n";

std::string InternalShaders::ssaoFragmentShader =
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





std::string InternalShaders::simpleLitVertexShader =
"uniform mat4 model;\n"
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

std::string InternalShaders::simpleLitFragmentShader =
"in vec3 FragPos;\n"
"in vec3 CameraPos;\n"
"in vec3 Normal;\n"
"in vec2 TexCoord;\n"

"out vec4 FragColor;\n"

"void main(void)\n"
"{\n"
"	FragColor = vec4(1.0, 0.5, 0.5, 1.0);\n"
"}\n";

//std::string InternalShaders::simpleLitFragmentShader =
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