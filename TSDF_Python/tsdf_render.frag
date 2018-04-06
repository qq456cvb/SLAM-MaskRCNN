#version 400
in vec2 vTexCoord;

uniform sampler2D tsdf;
uniform isampler2D tsdf_cnt;

out vec4 frag_color;

uniform mat4 s2w;
uniform vec3 c;

uniform float vol_dim;
uniform vec3 volStart;
uniform vec3 volEnd;

uniform vec3 random_colors[32];
// vec3 texToPos(vec2 texCoord, vec2 texel, float tex_dim, float voxel)
// {
//     vec3 pos;
//     float idx = floor(texCoord.y/texel.y) * tex_dim + floor(texCoord.x/texel.x);
//     float idxz = floor(idx/(vol_dim*vol_dim));
//     float idxxy = idx - idxz*(vol_dim*vol_dim);
//     float idxy = floor(idxxy/vol_dim);
//     float idxx = floor(idxxy - idxy*vol_dim);
//     pos = vec3(idxx, idxy, idxz) * voxel + volStart;
//     return pos;
// }

vec2 indToTex(vec3 ind, vec2 texel, float tex_dim) {
//    vec3 ind = (pos - volStart) / voxel;
    float tex_ind_1d = ind.x * vol_dim * vol_dim + ind.y * vol_dim + ind.z;
    float tex_ind_y = floor(tex_ind_1d / tex_dim);
    float tex_ind_x = tex_ind_1d - tex_dim * tex_ind_y;

    // add texel / 2 to adjust dicrete coordinates to continous one
    vec2 tex_ind = vec2(tex_ind_x, tex_ind_y) * texel + texel / 2;
    return tex_ind;
}

// TODO: utilize opengl's interpolation
vec4 interpTsdf(vec3 pos, vec2 texel, float tex_dim, float voxel) {
//    pos += 0.00001f;
    vec3 ind = (pos - volStart) / voxel;
    vec3 interp = fract(ind);
    ind = floor(ind);
    vec2 tex_lll = indToTex(ind, texel, tex_dim);
    vec2 tex_hll = indToTex(ind + vec3(1, 0, 0), texel, tex_dim);
    vec2 tex_lhl = indToTex(ind + vec3(0, 1, 0), texel, tex_dim);
    vec2 tex_hhl = indToTex(ind + vec3(1, 1, 0), texel, tex_dim);
    vec4 low = mix( mix(texture(tsdf, tex_lll), texture(tsdf, tex_hll), interp.x),  mix(texture(tsdf, tex_lhl), texture(tsdf, tex_hhl), interp.x), interp.y);

    vec2 tex_llh = indToTex(ind + vec3(0, 0, 1), texel, tex_dim);
    vec2 tex_hlh = indToTex(ind + vec3(1, 0, 1), texel, tex_dim);
    vec2 tex_lhh = indToTex(ind + vec3(0, 1, 1), texel, tex_dim);
    vec2 tex_hhh = indToTex(ind + vec3(1, 1, 1), texel, tex_dim);
    vec4 high = mix( mix(texture(tsdf, tex_llh), texture(tsdf, tex_hlh), interp.x),  mix(texture(tsdf, tex_lhh), texture(tsdf, tex_hhh), interp.x), interp.y);
    return mix(low, high, interp.z);
}

vec4 interpTsdfCnt(vec3 pos, vec2 texel, float tex_dim, float voxel) {
//    pos += 0.00001f;
    vec3 ind = (pos - volStart) / voxel;
    vec3 interp = fract(ind);
    ind = floor(ind);
    vec2 tex_lll = indToTex(ind, texel, tex_dim);
    vec2 tex_hll = indToTex(ind + vec3(1, 0, 0), texel, tex_dim);
    vec2 tex_lhl = indToTex(ind + vec3(0, 1, 0), texel, tex_dim);
    vec2 tex_hhl = indToTex(ind + vec3(1, 1, 0), texel, tex_dim);
    vec4 low = mix( mix(texture(tsdf_cnt, tex_lll), texture(tsdf_cnt, tex_hll), interp.x),  mix(texture(tsdf_cnt, tex_lhl), texture(tsdf_cnt, tex_hhl), interp.x), interp.y);

    vec2 tex_llh = indToTex(ind + vec3(0, 0, 1), texel, tex_dim);
    vec2 tex_hlh = indToTex(ind + vec3(1, 0, 1), texel, tex_dim);
    vec2 tex_lhh = indToTex(ind + vec3(0, 1, 1), texel, tex_dim);
    vec2 tex_hhh = indToTex(ind + vec3(1, 1, 1), texel, tex_dim);
    vec4 high = mix( mix(texture(tsdf_cnt, tex_llh), texture(tsdf_cnt, tex_hlh), interp.x),  mix(texture(tsdf_cnt, tex_lhh), texture(tsdf_cnt, tex_hhh), interp.x), interp.y);
    return mix(low, high, interp.z);
}

void main(void)
{
//    frag_color = vec4(texture(tsdf, vTexCoord).bgr, 1);
    vec3 vol = volEnd - volStart;
    float mu = vol.x / vol_dim * 2.;
    float voxel = vol.x / (vol_dim - 1);
    float tex_dim = sqrt(vol_dim * vol_dim * vol_dim);
    vec2 texel = vec2(1.0/tex_dim);

    vec4 screen_pos = vec4(vec2(vTexCoord.x, vTexCoord.y) * vec2(640, 480), 1, 1);
    vec4 target = s2w * screen_pos;
    vec3 d = target.xyz - c;
    d = normalize(d);
    vec3 inv_d = 1. / d;
    vec3 tbot = inv_d * (volStart - c);
    vec3 ttop = inv_d * (volEnd - c);

    vec3 tmin = vec3(min(ttop.x, tbot.x), min(ttop.y, tbot.y), min(ttop.z, tbot.z));
    float tnear = max(max(tmin.x, tmin.y), tmin.z);
    tnear = max(tnear, 0.01);

    vec3 tmax = vec3(max(ttop.x, tbot.x), max(ttop.y, tbot.y), max(ttop.z, tbot.z));
    float tfar = min(min(tmax.x, tmax.y), tmax.z);
    tfar = min(tfar, 100);
    if (tnear > tfar) discard;
    float t = tnear;
    float stepsize = voxel;

    float f_t = interpTsdf(c + t * d, texel, tex_dim, voxel).a;
    float f_tt = 0;
    vec4 color = vec4(0, 0, 0, 1);


    if (f_t > 0) {
        for(; t < tfar; t += stepsize){
            f_tt = interpTsdf(c + t * d, texel, tex_dim, voxel).a;
            if(f_tt < 0.0)                               // got it, jump out of inner loop
            {
                break;
            }
            if(f_tt < voxel / 2)                            // coming closer, reduce stepsize
            {
                stepsize = voxel / 4;
            }
            f_t = f_tt;
        }
        if(f_tt < 0.0){                               // got it, calculate accurate intersection
            t = t + stepsize * f_tt / (f_t - f_tt);
            vec3 pt = c + t * d;
            // color.rgb = interpTsdf(pt, texel, tex_dim, voxel).rgb;
            color.rgb = random_colors[int(round(interpTsdfCnt(pt, texel, tex_dim, voxel).r))].rgb;
            // color = vec4(0, 0, 1, 1);
        }

    }
//    frag_color = texture(tsdf, vTexCoord);
//    color = texture(tsdf, vTexCoord);
//    frag_color = vec4(color.a / mu, 0, 0, 1);
//    if (color.a < 0) discard;
    frag_color = color;
}