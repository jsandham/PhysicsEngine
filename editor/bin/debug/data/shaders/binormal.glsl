#vertex
#version 430 core
layout(std140) uniform CameraBlock
{
    mat4 projection;
    mat4 view;
    mat4 viewProjection;
    vec3 cameraPos;
}Camera;
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;

uniform mat4 model;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
void main()
{
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = normalize(normal);
    TexCoord = texCoord;
    gl_Position = Camera.projection * Camera.view * vec4(FragPos, 1.0);
}

#fragment
#version 430 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
out vec4 FragColor;
void main(void)
{
    // derivations of the fragment position
    vec3 pos_dx = dFdx(FragPos);
    vec3 pos_dy = dFdy(FragPos);
    // derivations of the texture coordinate
    vec2 texC_dx = dFdx(TexCoord);
    vec2 texC_dy = dFdy(TexCoord);
    // tangent vector and binormal vector
    vec3 tangent = texC_dy.y * pos_dx - texC_dx.y * pos_dy;
    vec3 binormal = texC_dx.x * pos_dy - texC_dy.x * pos_dx;
    tangent = tangent - Normal * dot(tangent, Normal);
    binormal = binormal - Normal * dot(binormal, Normal);
    binormal = binormal - tangent * dot(binormal, tangent);
    FragColor = vec4(normalize(binormal), 1.0f);
}
