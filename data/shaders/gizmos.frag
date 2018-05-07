#version 330 core

in vec4 Color;

out vec4 FragPos;

void main()
{
	FragPos = Color;	
}