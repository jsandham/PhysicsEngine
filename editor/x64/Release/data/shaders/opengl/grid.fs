#version 430 core
in vec4 Color;
out vec4 FragColor;
void main()
{
	float depth = 0.2f * gl_FragCoord.z / gl_FragCoord.w;
	FragColor = vec4(Color.x, Color.y, Color.z, clamp(1.0f / depth, 0.0f, 0.8f));
}
