VERTEX:

#version 330 core

layout (std140) uniform CameraBlock
{
	mat4 projection;
	mat4 view;
	vec3 cameraPos;
}camera;


uniform mat4 model;

in vec3 position;
in vec3 normal;
in vec2 texCoord;

out vec3 Normal;
out vec2 TexCoord;

void main()
{
    gl_Position = camera.projection * camera.view * model * vec4(position, 1.0);

    Normal = normal;
    TexCoord = texCoord;
}





FRAGMENT:

#version 330 core

uniform sampler2D mainTexture;

in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

void main(void) {
	FragColor = texture(mainTexture, TexCoord);
    //FragColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);
    //FragColor = vec4(Normal.x, Normal.y, Normal.z, 1.0f);
}