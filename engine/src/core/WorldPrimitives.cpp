#include "../../include/core/WorldPrimitives.h"
#include "../../include/core/World.h"

#include "Windows.h"

#include <set>
#include <glm/glm.hpp>

using namespace PhysicsEngine;

void WorldPrimitives::createPrimitiveMeshes(World* world, int nx, int nz)
{
    Mesh *plane = world->createAsset<Mesh>(Guid("83a08619-90c8-49a0-af38-9fb3732c6bc3"));
    Mesh *disc = world->createAsset<Mesh>(Guid("a564ec96-bd6f-493a-ad2a-0009a7cb0fb0"));
    Mesh *cube = world->createAsset<Mesh>(Guid("94f5dcfc-977b-44ad-ba7c-502ce049a187"));
    Mesh *sphere = world->createAsset<Mesh>(Guid("6b415ea5-1c19-4d0d-a6df-62461167b4b3"));
    Mesh *cylinder = world->createAsset<Mesh>(Guid("af6181eb-1b8e-4102-9b6b-39ba470a87e9"));
    Mesh *cone = world->createAsset<Mesh>(Guid("d03432f9-4f06-436f-8c2f-8d3d1264da25"));

    assert(plane != nullptr);
    assert(disc != nullptr);
    assert(cube != nullptr);
    assert(sphere != nullptr);
    assert(cylinder != nullptr);
    assert(cone != nullptr);

    Shader *standardShader = world->createAsset<Shader>(Guid("d875cfa8-d25b-4c5d-a26d-caafd191baf7"));
    
    assert(standardShader != nullptr);

    // TODO: Move to opengl file and using switch on something getAPI() to get shader in api independednt way
    std::string vertexShader =
        "#version 430 core\n"
        "layout(std140) uniform CameraBlock\n"
        "{\n"
        "    mat4 projection;\n"
        "    mat4 view;\n"
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
        "uniform mat4 model;\n"
        "in vec3 position;\n"
        "in vec3 normal;\n"
        "in vec2 texCoord;\n"
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
        "    gl_Position = Camera.projection * Camera.view * vec4(FragPos, 1.0);\n"
        "    ClipSpaceZ = gl_Position.z;\n"
        "    for (int i = 0; i < 5; i++)\n"
        "    {\n"
        "        FragPosLightSpace[i] = Light.lightProjection[i] * Light.lightView[i] * vec4(FragPos, 1.0f);\n"
        "    }\n"
        "}\n";

    std::string fragmentShader = "#version 430 core\n"
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
                           "    sampler2D mainTexture;\n"
                           "    sampler2D normalMap;\n"
                           "    sampler2D specularMap;\n"
                           "};\n"
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
                           "#if defined (DIRECTIONALLIGHT)\n"
                           "    FragColor = vec4(CalcDirLight(material, Normal, viewDir), 1.0f) * texture(material.mainTexture, TexCoord);\n"
                           "#elif defined(SPOTLIGHT)\n"
                           "    FragColor = vec4(CalcSpotLight(material, Normal, FragPos, viewDir), 1.0f) * texture(material.mainTexture, TexCoord);\n"
                           "#elif defined(POINTLIGHT)\n"
                           "    FragColor = vec4(CalcPointLight(material, Normal, FragPos, viewDir), 1.0f) * texture(material.mainTexture, TexCoord);\n"
                           "#else\n"
                           "    FragColor = vec4(0.5, 0.5, 0.5, 1.0);\n"
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
                           "}\n";

    standardShader->setName("Standard");
    standardShader->setVertexShader(vertexShader);
    standardShader->setFragmentShader(fragmentShader);

    std::set<ShaderMacro> variant0;
    variant0.insert(ShaderMacro::None);

    std::set<ShaderMacro> variant1;
    variant1.insert(ShaderMacro::Directional);
    variant1.insert(ShaderMacro::HardShadows);
    variant1.insert(ShaderMacro::ShowCascades);

    std::set<ShaderMacro> variant2;
    variant2.insert(ShaderMacro::Directional);
    variant2.insert(ShaderMacro::SoftShadows);
    variant2.insert(ShaderMacro::ShowCascades);

    std::set<ShaderMacro> variant3;
    variant3.insert(ShaderMacro::Spot);
    variant3.insert(ShaderMacro::HardShadows);

    std::set<ShaderMacro> variant4;
    variant4.insert(ShaderMacro::Spot);
    variant4.insert(ShaderMacro::SoftShadows);

    std::set<ShaderMacro> variant5;
    variant5.insert(ShaderMacro::Point);
    variant5.insert(ShaderMacro::HardShadows);

    std::set<ShaderMacro> variant6;
    variant6.insert(ShaderMacro::Point);
    variant6.insert(ShaderMacro::SoftShadows);

    standardShader->addVariant(0, variant0);
    standardShader->addVariant(1, variant1);
    standardShader->addVariant(2, variant2);
    standardShader->addVariant(3, variant3);
    standardShader->addVariant(4, variant4);
    standardShader->addVariant(5, variant5);
    standardShader->addVariant(6, variant6);

    Material *standardMaterial = world->createAsset<Material>(Guid("1d83f0b2-f16d-48e6-9cbd-20be8115179b"));
    
    assert(standardMaterial != nullptr);

    standardMaterial->setName("Standard");
    standardMaterial->setShaderId(standardShader->getId());

    //.  .  .  .  .  .  .  .  .  .
    //.  .  .  .  .  .  .  .  .  .
    //.  .  .  .  .  .  .  .  .  .
    //.  .  .  .  .  .  .  .  .  .

    // Generate plane mesh
    int triangleCount = 2 * (nx - 1) * (nz - 1);
    int vertexCount = 3 * triangleCount;

    std::vector<float> planeVertices(3 * vertexCount);
    std::vector<float> planeNormals(3 * vertexCount);
    std::vector<float> planeTexCoords(2 * vertexCount);

    float xmin = -0.5f;
    float xmax = 0.5f;
    float zmin = -0.5f;
    float zmax = 0.5f;

    float dx = (xmax - xmin) / (nx - 1);
    float dz = (zmax - zmin) / (nz - 1);

    int i = 0;
    int j = 0;
    int k = 0;
    for (int z = 0; z < (nz - 1); z++)
    {
        for (int x = 0; x < (nx - 1); x++)
        {
            // first triangle
            planeVertices[i++] = xmin + x * dx;
            planeVertices[i++] = 0.0f;
            planeVertices[i++] = zmin + z * dz;
            
            planeVertices[i++] = xmin + (x + 1) * dx;
            planeVertices[i++] = 0.0f;
            planeVertices[i++] = zmin + z * dz;

            planeVertices[i++] = xmin + x * dx;
            planeVertices[i++] = 0.0f;
            planeVertices[i++] = zmin + (z + 1) * dz;

            planeTexCoords[k++] = x * dx;
            planeTexCoords[k++] = z * dz;

            planeTexCoords[k++] = (x + 1) * dx;
            planeTexCoords[k++] = z * dz;

            planeTexCoords[k++] = x * dx;
            planeTexCoords[k++] = (z + 1) * dz;

            // second triangle
            planeVertices[i++] = xmin + (x + 1) * dx;
            planeVertices[i++] = 0.0f;
            planeVertices[i++] = zmin + z * dz;

            planeVertices[i++] = xmin + (x + 1) * dx;
            planeVertices[i++] = 0.0f;
            planeVertices[i++] = zmin + (z + 1) * dz;

            planeVertices[i++] = xmin + x * dx;
            planeVertices[i++] = 0.0f;
            planeVertices[i++] = zmin + (z + 1) * dz;

            planeTexCoords[k++] = (x + 1) * dx;
            planeTexCoords[k++] = z * dz;

            planeTexCoords[k++] = (x + 1) * dx;
            planeTexCoords[k++] = (z + 1) * dz;

            planeTexCoords[k++] = x * dx;
            planeTexCoords[k++] = (z + 1) * dz;

            for (int n = 0; n < 6; n++)
            {
                planeNormals[j++] = 0.0f;
                planeNormals[j++] = 1.0f;
                planeNormals[j++] = 0.0f;
            }
        }
    }

    plane->load(planeVertices, planeNormals, planeTexCoords, {0, 3 * vertexCount});

    // Generate disc
    int discTriangleCount = nx * nz;
    int discVertexCount = 3 * discTriangleCount;

    std::vector<float> discVertices(3 * discVertexCount);
    std::vector<float> discNormals(3 * discVertexCount);
    std::vector<float> discTexCoords(2 * discVertexCount);

    float dtheta = 360.0f / discTriangleCount;

    i = 0;
    j = 0;
    k = 0;
    for (int s = 0; s < discTriangleCount; s++)
    {
        float theta0 = glm::radians(dtheta * s);
        float theta1 = glm::radians(dtheta * (s + 1));

        discVertices[i++] = 0.5f * glm::cos(theta0);
        discVertices[i++] = 0.0f;
        discVertices[i++] = 0.5f * glm::sin(theta0);

        discVertices[i++] = 0.5f * glm::cos(theta1);
        discVertices[i++] = 0.0f;
        discVertices[i++] = 0.5f * glm::sin(theta1);

        discVertices[i++] = 0.0f;
        discVertices[i++] = 0.0f;
        discVertices[i++] = 0.0f;

        for (int n = 0; n < 3; n++)
        {
            discNormals[j++] = 0.0f;
            discNormals[j++] = 1.0f;
            discNormals[j++] = 0.0f;
        }

        discTexCoords[k++] = 0.5f * glm::cos(theta0) + 0.5f;
        discTexCoords[k++] = 0.5f * glm::sin(theta0) + 0.5f;

        discTexCoords[k++] = 0.5f * glm::cos(theta1) + 0.5f;
        discTexCoords[k++] = 0.5f * glm::sin(theta1) + 0.5f;

        discTexCoords[k++] = 0.5f;
        discTexCoords[k++] = 0.5f;
    }

    disc->load(discVertices, discNormals, discTexCoords, {0, 3 * discVertexCount});

    // Generate cube mesh
    int cubeVertexCount = 6 * vertexCount;

    std::vector<float> cubeVertices(3 * cubeVertexCount);
    std::vector<float> cubeNormals(3 * cubeVertexCount);
    std::vector<float> cubeTexCoords(2 * cubeVertexCount);
    
    int faceStart = 0;
    int texStart = 0;

    // bottom face
    for (int v = 0; v < vertexCount; v++)
    {
        cubeVertices[3 * v + 0 + faceStart] = planeVertices[3 * v + 0];
        cubeVertices[3 * v + 1 + faceStart] = -0.5f;
        cubeVertices[3 * v + 2 + faceStart] = planeVertices[3 * v + 2];

        cubeNormals[3 * v + 0 + faceStart] = 0.0f;
        cubeNormals[3 * v + 1 + faceStart] = -1.0f;
        cubeNormals[3 * v + 2 + faceStart] = 0.0f;

        cubeTexCoords[2 * v + 0 + texStart] = planeTexCoords[2 * v + 0];
        cubeTexCoords[2 * v + 1 + texStart] = planeTexCoords[2 * v + 1];

    }
    faceStart += 3 * vertexCount;
    texStart += 2 * vertexCount;

    // top face
    for (int v = 0; v < vertexCount; v++)
    {
        cubeVertices[3 * v + 0 + faceStart] = planeVertices[3 * v + 0];
        cubeVertices[3 * v + 1 + faceStart] = 0.5f;
        cubeVertices[3 * v + 2 + faceStart] = planeVertices[3 * v + 2];

        cubeNormals[3 * v + 0 + faceStart] = 0.0f;
        cubeNormals[3 * v + 1 + faceStart] = 1.0f;
        cubeNormals[3 * v + 2 + faceStart] = 0.0f;

        cubeTexCoords[2 * v + 0 + texStart] = planeTexCoords[2 * v + 0];
        cubeTexCoords[2 * v + 1 + texStart] = planeTexCoords[2 * v + 1];
    }
    faceStart += 3 * vertexCount;
    texStart += 2 * vertexCount;

    // left face
    for (int v = 0; v < vertexCount; v++)
    {
        cubeVertices[3 * v + 0 + faceStart] = -0.5f;
        cubeVertices[3 * v + 1 + faceStart] = planeVertices[3 * v + 0];
        cubeVertices[3 * v + 2 + faceStart] = planeVertices[3 * v + 2];

        cubeNormals[3 * v + 0 + faceStart] = -1.0f;
        cubeNormals[3 * v + 1 + faceStart] = 0.0f;
        cubeNormals[3 * v + 2 + faceStart] = 0.0f;

        cubeTexCoords[2 * v + 0 + texStart] = planeTexCoords[2 * v + 0];
        cubeTexCoords[2 * v + 1 + texStart] = planeTexCoords[2 * v + 1];
    }
    faceStart += 3 * vertexCount;
    texStart += 2 * vertexCount;

    // right face
    for (int v = 0; v < vertexCount; v++)
    {
        cubeVertices[3 * v + 0 + faceStart] = 0.5f;
        cubeVertices[3 * v + 1 + faceStart] = planeVertices[3 * v + 0];
        cubeVertices[3 * v + 2 + faceStart] = planeVertices[3 * v + 2];

        cubeNormals[3 * v + 0 + faceStart] = 1.0f;
        cubeNormals[3 * v + 1 + faceStart] = 0.0f;
        cubeNormals[3 * v + 2 + faceStart] = 0.0f;

        cubeTexCoords[2 * v + 0 + texStart] = planeTexCoords[2 * v + 0];
        cubeTexCoords[2 * v + 1 + texStart] = planeTexCoords[2 * v + 1];
    }
    faceStart += 3 * vertexCount;
    texStart += 2 * vertexCount;

    // near face
    for (int v = 0; v < vertexCount; v++)
    {
        cubeVertices[3 * v + 0 + faceStart] = planeVertices[3 * v + 2];
        cubeVertices[3 * v + 1 + faceStart] = planeVertices[3 * v + 0];
        cubeVertices[3 * v + 2 + faceStart] = -0.5;

        cubeNormals[3 * v + 0 + faceStart] = 0.0f;
        cubeNormals[3 * v + 1 + faceStart] = 0.0f;
        cubeNormals[3 * v + 2 + faceStart] = -1.0f;

        cubeTexCoords[2 * v + 0 + texStart] = planeTexCoords[2 * v + 0];
        cubeTexCoords[2 * v + 1 + texStart] = planeTexCoords[2 * v + 1];
    }
    faceStart += 3 * vertexCount;
    texStart += 2 * vertexCount;

    // far face
    for (int v = 0; v < vertexCount; v++)
    {
        cubeVertices[3 * v + 0 + faceStart] = planeVertices[3 * v + 2];
        cubeVertices[3 * v + 1 + faceStart] = planeVertices[3 * v + 0];
        cubeVertices[3 * v + 2 + faceStart] = 0.5f;

        cubeNormals[3 * v + 0 + faceStart] = 0.0f;
        cubeNormals[3 * v + 1 + faceStart] = 0.0f;
        cubeNormals[3 * v + 2 + faceStart] = 1.0f;

        cubeTexCoords[2 * v + 0 + texStart] = planeTexCoords[2 * v + 0];
        cubeTexCoords[2 * v + 1 + texStart] = planeTexCoords[2 * v + 1];
    }
    faceStart += 3 * vertexCount;
    texStart += 2 * vertexCount;

    cube->load(cubeVertices, cubeNormals, cubeTexCoords, {0, 3 * cubeVertexCount});

    // Generate Sphere mesh
    int sphereVertexCount = cubeVertexCount;

    std::vector<float> sphereVertices(3 * sphereVertexCount);
    std::vector<float> sphereNormals(3 * sphereVertexCount);
    std::vector<float> sphereTexCoords(2 * sphereVertexCount);

    for (int v = 0; v < sphereVertexCount; v++)
    {
        float vx = cubeVertices[3 * v + 0];
        float vy = cubeVertices[3 * v + 1];
        float vz = cubeVertices[3 * v + 2];

        float length = glm::sqrt(vx * vx + vy * vy + vz * vz);

        vx = vx / length;
        vy = vy / length;
        vz = vz / length;
        
        sphereVertices[3 * v + 0] = vx;
        sphereVertices[3 * v + 1] = vy;
        sphereVertices[3 * v + 2] = vz;
            
        sphereNormals[3 * v + 0] = vx;
        sphereNormals[3 * v + 1] = vy;
        sphereNormals[3 * v + 2] = vz;

        sphereTexCoords[2 * v + 0] = cubeTexCoords[2 * v + 0];
        sphereTexCoords[2 * v + 1] = cubeTexCoords[2 * v + 1];
    }

    sphere->load(sphereVertices, sphereNormals, sphereTexCoords, {0, 3 * sphereVertexCount});

    // Generate cylinder
    int cylinderTriangleCount = 4 * discTriangleCount;
    int cylinderVertexCount = 3 * cylinderTriangleCount;
    
    std::vector<float> cylinderVertices(3 * cylinderVertexCount);
    std::vector<float> cylinderNormals(3 * cylinderVertexCount);
    std::vector<float> cylinderTexCoords(3 * cylinderVertexCount);

    faceStart = 0;
    texStart = 0;

    // bottom
    for (int v = 0; v < discVertexCount; v++)
    {
        cylinderVertices[3 * v + 0 + faceStart] = discVertices[3 * v + 0];
        cylinderVertices[3 * v + 1 + faceStart] = -0.5f;
        cylinderVertices[3 * v + 2 + faceStart] = discVertices[3 * v + 2];

        cylinderNormals[3 * v + 0 + faceStart] = 0.0f;
        cylinderNormals[3 * v + 1 + faceStart] = -1.0f;
        cylinderNormals[3 * v + 2 + faceStart] = 0.0f;

        cylinderTexCoords[2 * v + 0 + texStart] = discTexCoords[2 * v + 0];
        cylinderTexCoords[2 * v + 1 + texStart] = discTexCoords[2 * v + 1];
    }

    faceStart += 3 * discVertexCount;
    texStart += 2 * discVertexCount;

    // top
    for (int v = 0; v < discVertexCount; v++)
    {
        cylinderVertices[3 * v + 0 + faceStart] = discVertices[3 * v + 0];
        cylinderVertices[3 * v + 1 + faceStart] = 0.5f;
        cylinderVertices[3 * v + 2 + faceStart] = discVertices[3 * v + 2];

        cylinderNormals[3 * v + 0 + faceStart] = 0.0f;
        cylinderNormals[3 * v + 1 + faceStart] = 1.0f;
        cylinderNormals[3 * v + 2 + faceStart] = 0.0f;

        cylinderTexCoords[2 * v + 0 + texStart] = discTexCoords[2 * v + 0];
        cylinderTexCoords[2 * v + 1 + texStart] = discTexCoords[2 * v + 1];
    }

    faceStart += 3 * discVertexCount;
    texStart += 2 * discVertexCount;

    // tube
    i = faceStart;
    j = faceStart;
    k = texStart;
    /*for (int v = 0; v < discVertexCount; v++)
    {
        // skip center disc vertex
        if ((v + 1) % 3 != 0)
        {
            // first triangle
            cylinderVertices[i++] = discVertices[3 * v + 0];
            cylinderVertices[i++] = -0.5f;
            cylinderVertices[i++] = discVertices[3 * v + 2];

            cylinderVertices[i++] = discVertices[3 * (v + 1) + 0];
            cylinderVertices[i++] = -0.5f;
            cylinderVertices[i++] = discVertices[3 * (v + 1) + 2];

            cylinderVertices[i++] = discVertices[3 * (v + 1) + 0];
            cylinderVertices[i++] = 0.5f;
            cylinderVertices[i++] = discVertices[3 * (v + 1) + 2];

            cylinderNormals[j++] = discVertices[3 * v + 0];
            cylinderNormals[j++] = 0.0f;
            cylinderNormals[j++] = discVertices[3 * v + 2];

            cylinderNormals[j++] = discVertices[3 * (v + 1) + 0];
            cylinderNormals[j++] = 0.0f;
            cylinderNormals[j++] = discVertices[3 * (v + 1) + 2];

            cylinderNormals[j++] = discVertices[3 * (v + 1) + 0];
            cylinderNormals[j++] = 0.0f;
            cylinderNormals[j++] = discVertices[3 * (v + 1) + 2];

            // second triangle
            cylinderVertices[i++] = discVertices[3 * v + 0];
            cylinderVertices[i++] = -0.5f;
            cylinderVertices[i++] = discVertices[3 * v + 2];

            cylinderVertices[i++] = discVertices[3 * (v + 1) + 0];
            cylinderVertices[i++] = 0.5f;
            cylinderVertices[i++] = discVertices[3 * (v + 1) + 2];

            cylinderVertices[i++] = discVertices[3 * v + 0];
            cylinderVertices[i++] = 0.5f;
            cylinderVertices[i++] = discVertices[3 * v + 2];

            cylinderNormals[j++] = discVertices[3 * v + 0];
            cylinderNormals[j++] = 0.0f;
            cylinderNormals[j++] = discVertices[3 * v + 2];

            cylinderNormals[j++] = discVertices[3 * (v + 1) + 0];
            cylinderNormals[j++] = 0.0f;
            cylinderNormals[j++] = discVertices[3 * (v + 1) + 2];

            cylinderNormals[j++] = discVertices[3 * v + 0];
            cylinderNormals[j++] = 0.0f;
            cylinderNormals[j++] = discVertices[3 * v + 2];
        }
    }*/

    cylinder->load(cylinderVertices, cylinderNormals, cylinderTexCoords, {0, 3 * cylinderVertexCount});

    mPlaneMeshId = plane->getId();
    mDiscMeshId = disc->getId();
    mCubeMeshId = cube->getId();
    mSphereMeshId = sphere->getId();
    mCylinderMeshId = cylinder->getId();
    mConeMeshId = cone->getId();

    mStandardShaderId = standardShader->getId();
    mStandardMaterialId = standardMaterial->getId();
}