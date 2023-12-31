struct PS_INPUT
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

Texture2D screenTexture;
SamplerState sampleState
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

// pixel shader
float4 PSMain(PS_INPUT input) : SV_TARGET
{
    float3 col = screenTexture.Sample(sampleState, input.texCoord).rgb;
    return float4(col, 1.0f);
}