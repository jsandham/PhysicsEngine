// INPUT/OUTPUT structures
struct VS_INPUT
{
    float3 position : POSITION;
    float3 normal : NORMAL;
};

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
};

// constant buffers
cbuffer CAMERA_CONSTANT_BUFFER : register(b0)
{
    matrix projection;
    matrix view;
    matrix viewProjection;
    float3 cameraPos;
}

cbuffer VS_CONSTANT_BUFFER2 : register(b3)
{
    matrix model;
}

// vertex shader
VS_OUTPUT VSMain(VS_INPUT input)
{
    float4x4 modelViewProj = mul(projection, mul(view, model));
    VS_OUTPUT output;
    output.position = mul(modelViewProj, float4(input.position, 1.0f));
    output.normal = input.normal;

    return output;
}