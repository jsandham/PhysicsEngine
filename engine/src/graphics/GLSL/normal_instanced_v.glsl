#version 430 core
#include "camera.glsl"
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 3) in mat4 model;

out vec3 Normal;
void main()
{
    vec4 worldPos = model * vec4(aPos, 1.0);
    mat3 normalMatrix = transpose(inverse(mat3(model)));
    Normal = normalMatrix * aNormal;
    gl_Position = Camera.projection * Camera.view * worldPos;
}
