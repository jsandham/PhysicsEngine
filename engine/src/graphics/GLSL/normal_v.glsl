#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(std140) uniform CameraBlock
{
    mat4 projection;
    mat4 view;
    vec3 cameraPos;
}Camera;
out vec3 Normal;
uniform mat4 model;
void main()
{
    vec4 worldPos = model * vec4(aPos, 1.0);
    mat3 normalMatrix = transpose(inverse(mat3(model)));
    Normal = normalMatrix * aNormal;
    gl_Position = Camera.projection * Camera.view * worldPos;
}
