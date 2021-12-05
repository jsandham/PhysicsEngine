#version 430 core
layout(location = 0) in vec3 aPos;
layout(std140) uniform CameraBlock
{
    mat4 projection;
    mat4 view;
    vec3 cameraPos;
}Camera;
out vec3 FragPos;
uniform mat4 model;
void main()
{
    vec4 worldPos = model * vec4(aPos, 1.0);
    FragPos = worldPos.xyz;
    gl_Position = Camera.projection * Camera.view * worldPos;
}