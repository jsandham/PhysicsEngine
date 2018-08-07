VERTEX:

#version 330 core

layout (std140) uniform CameraBlock
{
	mat4 projection;
	mat4 view;
	vec3 cameraPos;
};

uniform mat4 model;

in vec3 position;
in vec3 normal;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
}





FRAGMENT:

#version 330 core

out vec4 FragColor;

void main(void) {
    FragColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);
}