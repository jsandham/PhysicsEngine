#version 430 core
#include "camera.glsl"
layout(location = 0) in vec3 aPos;
layout(location = 3) in mat4 model;

out vec3 FragPos;
void main()
{
    vec4 worldPos = model * vec4(aPos, 1.0);
    FragPos = worldPos.xyz;
    gl_Position = Camera.viewProjection * worldPos;
}