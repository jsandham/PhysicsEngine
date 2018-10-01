#version 330 core

in vec3 position;
//in vec4 color;

out vec4 Color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec4 color;

void main()
{
	Color = color;

	gl_Position = projection * view * model * vec4(position, 1.0);
}