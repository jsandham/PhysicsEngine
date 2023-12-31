struct PS_INPUT
{
    float4 position : SV_POSITION;
};

struct Material
{
    uint4 color;
};

cbuffer VS_CONSTANT_BUFFER : register(b3)
{
    Material color;
}

// pixel shader
float4 PSMain(PS_INPUT input) : SV_TARGET
{
    return float4(color.color.r / 255.0f, color.color.g / 255.0f, color.color.b / 255.0f, color.color.a / 255.0f);
}