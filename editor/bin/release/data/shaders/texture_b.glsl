#vertex
#version 430 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texCoord;
out vec2 TexCoord;
void main()
{
	gl_Position = vec4(position, 0.0, 1.0);
	TexCoord = texCoord;
};

#fragment
#version 430 core
uniform sampler2D texture0;
in vec2 TexCoord;
out vec4 FragColor;
void main()
{
    FragColor = vec4(0, 0, texture(texture0, TexCoord).b, 1);
};