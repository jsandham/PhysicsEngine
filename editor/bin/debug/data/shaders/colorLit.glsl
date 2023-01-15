#vertex
#version 430 core
layout(std140) uniform CameraBlock
{
    mat4 projection;
    mat4 view;
    mat4 viewProjection;
    vec3 cameraPos;
}Camera;
uniform mat4 model;
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;
out vec3 FragPos;
out vec3 CameraPos;
out vec3 Normal;
out vec2 TexCoord;
void main()
{
    CameraPos = Camera.cameraPos;
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;
    TexCoord = texCoord;
    gl_Position = Camera.projection * Camera.view * vec4(FragPos, 1.0);
}

#fragment
#version 430 core
uniform vec3 lightDirection;
uniform vec3 color;
uniform int wireframe;
in vec3 FragPos;
in vec3 CameraPos;
in vec3 Normal;
in vec2 TexCoord;
out vec4 FragColor;
vec3 CalcDirLight(vec3 normal, vec3 viewDir);
void main(void)
{
    vec3 viewDir = normalize(CameraPos - FragPos);
    FragColor = vec4(CalcDirLight(Normal, viewDir) * color, 1.0f);
}
vec3 CalcDirLight(vec3 normal, vec3 viewDir)
{
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightDirection);
    vec3 reflectDir = reflect(-lightDir, norm);
    float diffuseStrength = max(dot(norm, lightDir), 0.0);
    float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0), 1.0f);
    vec3 ambient = vec3(0.7, 0.7, 0.7);
    vec3 diffuse = vec3(1.0, 1.0, 1.0) * diffuseStrength;
    vec3 specular = vec3(0.7, 0.7, 0.7) * specularStrength;
    return (ambient + diffuse + specular);
}
