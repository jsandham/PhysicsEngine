#version 430 core

in vec4 Color;
out vec4 FragColor;
void main()
{
    FragColor = vec4(Color.r / 255.0f, Color.g / 255.0f,
                      Color.b / 255.0f, Color.a / 255.0f);
}