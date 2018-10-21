#version 330 core

layout (std140) uniform CameraBlock
{
	mat4 projection;
	mat4 view;
	vec3 cameraPos;
};

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

in vec3 position;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
}