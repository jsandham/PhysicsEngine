#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 3) in mat4 model;
layout(location = 7) in uvec4 color;

out vec3 FragPos;
out vec3 Normal;
out vec4 Color;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    Color = vec4(color.r / 255.0f, color.g / 255.0f, color.b / 255.0f, color.a / 255.0f);

    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}