#version 430 core
#include "camera.glsl"
layout(location = 0) in vec3 aPos;

out vec3 FragPos;
uniform mat4 model;
void main()
{
    vec4 worldPos = model * vec4(aPos, 1.0);
    FragPos = worldPos.xyz;
    gl_Position = Camera.projection * Camera.view * worldPos;
}