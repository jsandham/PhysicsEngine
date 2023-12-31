// INPUT structures
struct PS_INPUT
{
    float4 position : SV_POSITION;
    float3 fragPos : FRAG_POSITION;
};

// pixel shader
float4 PSMain(PS_INPUT input) : SV_TARGET
{
    return float4(input.fragPos, 1.0f);
}