// INPUT/OUTPUT structures
struct VS_INPUT
{
    float3 position : POSITION;
};

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float4 color : COLOR;
};

// constant buffers
cbuffer VS_CONSTANT_BUFFER : register(b3)
{
    matrix mvp;
    float4 color;
}

// vertex shader
VS_OUTPUT VSMain(VS_INPUT input)
{
    VS_OUTPUT output;
    output.position = mul(mvp, float4(input.position, 1.0f));
    output.color = color;

    return output;
}