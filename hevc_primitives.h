/*****************************************************************************
Copyright@2015 MulticoreWare, Inc.  All Rights Reserved.

CONFIDENTIALITY:  This software source code is considered confidential
information.  It must be kept confidential in accordance with the terms
and conditions of your Software License Agreement
*****************************************************************************/

#include "common/uhd_common.h"

#include "hevc.h"

#include "hevc_bitdepth.h"
#include "hevc_algorithm.h"
#include <arm_neon.h>
#include <stdlib.h>

#define CLIP3(x, min, max) (((x) > max) ? max : (((x) < min) ? min : (x)))

#define CLIP_U8(x) CLIP3((x), 0, 255)
#define CLIP_S16(x) CLIP3((x), -32768, 32767)

static inline void FUNC(uhd_emulated_edge_mc)(uint8_t *buf, const uint8_t *src,
                                              ptrdiff_t buf_linesize,
                                              ptrdiff_t src_linesize,
                                              int block_w, int block_h,
                                              int src_x, int src_y, int w, int h)
{
    int x, y;
    int start_y, start_x, end_y, end_x;

    if (!w || !h)
    {
        return;
    }

    assert(block_w * sizeof(pixel) <= (int)UHDABS(buf_linesize));

    if (src_y >= h)
    {
        src -= src_y * src_linesize;
        src += (h - 1) * src_linesize;
        src_y = h - 1;
    }
    else if (src_y <= -block_h)
    {
        src -= src_y * src_linesize;
        src += (1 - block_h) * src_linesize;
        src_y = 1 - block_h;
    }
    if (src_x >= w)
    {
        src += (w - 1 - src_x) * sizeof(pixel);
        src_x = w - 1;
    }
    else if (src_x <= -block_w)
    {
        src += (1 - block_w - src_x) * sizeof(pixel);
        src_x = 1 - block_w;
    }

    start_y = UHDMAX(0, -src_y);
    start_x = UHDMAX(0, -src_x);
    end_y = UHDMIN(block_h, h - src_y);
    end_x = UHDMIN(block_w, w - src_x);
    assert(start_y < end_y && block_h);
    assert(start_x < end_x && block_w);

    w = end_x - start_x;
    src += start_y * src_linesize + start_x * sizeof(pixel);
    buf += start_x * sizeof(pixel);

    for (y = 0; y < start_y; y++)
    {
        memcpy(buf, src, w * sizeof(pixel));
        buf += buf_linesize;
    }

    for (; y < end_y; y++)
    {
        memcpy(buf, src, w * sizeof(pixel));
        src += src_linesize;
        buf += buf_linesize;
    }

    src -= src_linesize;
    for (; y < block_h; y++)
    {
        memcpy(buf, src, w * sizeof(pixel));
        buf += buf_linesize;
    }

    buf -= block_h * buf_linesize + start_x * sizeof(pixel);
    while (block_h--)
    {
        pixel *bufp = (pixel *)buf;

        for (x = 0; x < start_x; x++)
        {
            bufp[x] = bufp[start_x];
        }

        for (x = end_x; x < block_w; x++)
        {
            bufp[x] = bufp[end_x - 1];
        }
        buf += buf_linesize;
    }
}

#if BIT_DEPTH < 16
static void FUNC(put_pcm)(uint8_t *_dst, ptrdiff_t stride, int width, int height,
                          GetBitContext *gb, int pcm_bit_depth)
{
    int x, y;
    pixel *dst = (pixel *)_dst;

    stride /= sizeof(pixel);

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = get_bits(gb, pcm_bit_depth) << (BIT_DEPTH - pcm_bit_depth);
        }
        dst += stride;
    }
}

static uhd_always_inline void FUNC(transquant_bypass)(uint8_t *_dst, int16_t *coeffs,
                                                      ptrdiff_t stride, int size)
{
    int x, y;
    pixel *dst = (pixel *)_dst;

    stride /= sizeof(pixel);

    for (y = 0; y < size; y++)
    {
        for (x = 0; x < size; x++)
        {
            dst[x] = uhd_clip_pixel(dst[x] + *coeffs);
            coeffs++;
        }
        dst += stride;
    }
}

static void FUNC(transform_add4x4)(uint8_t *_dst, int16_t *coeffs,
                                   ptrdiff_t stride)
{
    FUNC(transquant_bypass)
    (_dst, coeffs, stride, 4);
}

static void FUNC(transform_add8x8)(uint8_t *_dst, int16_t *coeffs,
                                   ptrdiff_t stride)
{
    FUNC(transquant_bypass)
    (_dst, coeffs, stride, 8);
}

static void FUNC(transform_add16x16)(uint8_t *_dst, int16_t *coeffs,
                                     ptrdiff_t stride)
{
    FUNC(transquant_bypass)
    (_dst, coeffs, stride, 16);
}

static void FUNC(transform_add32x32)(uint8_t *_dst, int16_t *coeffs,
                                     ptrdiff_t stride)
{
    FUNC(transquant_bypass)
    (_dst, coeffs, stride, 32);
}

static void FUNC(transform_rdpcm)(int16_t *_coeffs, int16_t log2_size, int mode)
{
    int16_t *coeffs = (int16_t *)_coeffs;
    int x, y;
    int size = 1 << log2_size;

    if (mode)
    {
        coeffs += size;
        for (y = 0; y < size - 1; y++)
        {
            for (x = 0; x < size; x++)
            {
                coeffs[x] += coeffs[x - size];
            }
            coeffs += size;
        }
    }
    else
    {
        for (y = 0; y < size; y++)
        {
            for (x = 1; x < size; x++)
            {
                coeffs[x] += coeffs[x - 1];
            }
            coeffs += size;
        }
    }
}

static void FUNC(transform_skip)(int16_t *_coeffs, int16_t log2_size)
{
    int shift = 15 - BIT_DEPTH - log2_size;
    int x, y;
    int size = 1 << log2_size;
    int16_t *coeffs = _coeffs;

    if (shift > 0)
    {
        int offset = 1 << (shift - 1);
        for (y = 0; y < size; y++)
        {
            for (x = 0; x < size; x++)
            {
                *coeffs = (*coeffs + offset) >> shift;
                coeffs++;
            }
        }
    }
    else
    {
        for (y = 0; y < size; y++)
        {
            for (x = 0; x < size; x++)
            {
                *coeffs = *coeffs << -shift;
                coeffs++;
            }
        }
    }
}

#define SET(dst, x) (dst) = (x)
#define SCALE(dst, x) (dst) = uhd_clip_int16(((x) + add) >> shift)
#define ADD_AND_SCALE(dst, x) \
    (dst) = uhd_clip_pixel((dst) + uhd_clip_int16(((x) + add) >> shift))

#define TR_4x4_LUMA(dst, src, step, assign)            \
    do                                                 \
    {                                                  \
        int c0 = src[0 * step] + src[2 * step];        \
        int c1 = src[2 * step] + src[3 * step];        \
        int c2 = src[0 * step] - src[3 * step];        \
        int c3 = 74 * src[1 * step];                   \
                                                       \
        assign(dst[2 * step], 74 * (src[0 * step] -    \
                                    src[2 * step] +    \
                                    src[3 * step]));   \
        assign(dst[0 * step], 29 * c0 + 55 * c1 + c3); \
        assign(dst[1 * step], 55 * c2 - 29 * c1 + c3); \
        assign(dst[3 * step], 55 * c0 + 29 * c2 - c3); \
    } while (0)

static void FUNC(transform_4x4_luma)(int16_t *coeffs)
{
    int i;
    int shift = 7;
    int add = 1 << (shift - 1);
    int16_t *src = coeffs;

    for (i = 0; i < 4; i++)
    {
        TR_4x4_LUMA(src, src, 4, SCALE);
        src++;
    }

    shift = 20 - BIT_DEPTH;
    add = 1 << (shift - 1);
    for (i = 0; i < 4; i++)
    {
        TR_4x4_LUMA(coeffs, coeffs, 1, SCALE);
        coeffs += 4;
    }
}

#undef TR_4x4_LUMA

#define TR_4(dst, src, dstep, sstep, assign, end)                 \
    do                                                            \
    {                                                             \
        const int e0 = 64 * src[0 * sstep] + 64 * src[2 * sstep]; \
        const int e1 = 64 * src[0 * sstep] - 64 * src[2 * sstep]; \
        const int o0 = 83 * src[1 * sstep] + 36 * src[3 * sstep]; \
        const int o1 = 36 * src[1 * sstep] - 83 * src[3 * sstep]; \
                                                                  \
        assign(dst[0 * dstep], e0 + o0);                          \
        assign(dst[1 * dstep], e1 + o1);                          \
        assign(dst[2 * dstep], e1 - o1);                          \
        assign(dst[3 * dstep], e0 - o0);                          \
    } while (0)

#define TR_8(dst, src, dstep, sstep, assign, end)               \
    do                                                          \
    {                                                           \
        int i, j;                                               \
        int e_8[4];                                             \
        int o_8[4] = {0};                                       \
        for (i = 0; i < 4; i++)                                 \
            for (j = 1; j < end; j += 2)                        \
                o_8[i] += transform[4 * j][i] * src[j * sstep]; \
        TR_4(e_8, src, 1, 2 * sstep, SET, 4);                   \
                                                                \
        for (i = 0; i < 4; i++)                                 \
        {                                                       \
            assign(dst[i * dstep], e_8[i] + o_8[i]);            \
            assign(dst[(7 - i) * dstep], e_8[i] - o_8[i]);      \
        }                                                       \
    } while (0)

#define TR_16(dst, src, dstep, sstep, assign, end)               \
    do                                                           \
    {                                                            \
        int i, j;                                                \
        int e_16[8];                                             \
        int o_16[8] = {0};                                       \
        for (i = 0; i < 8; i++)                                  \
            for (j = 1; j < end; j += 2)                         \
                o_16[i] += transform[2 * j][i] * src[j * sstep]; \
        TR_8(e_16, src, 1, 2 * sstep, SET, 8);                   \
                                                                 \
        for (i = 0; i < 8; i++)                                  \
        {                                                        \
            assign(dst[i * dstep], e_16[i] + o_16[i]);           \
            assign(dst[(15 - i) * dstep], e_16[i] - o_16[i]);    \
        }                                                        \
    } while (0)

#define TR_32(dst, src, dstep, sstep, assign, end)            \
    do                                                        \
    {                                                         \
        int i, j;                                             \
        int e_32[16];                                         \
        int o_32[16] = {0};                                   \
        for (i = 0; i < 16; i++)                              \
            for (j = 1; j < end; j += 2)                      \
                o_32[i] += transform[j][i] * src[j * sstep];  \
        TR_16(e_32, src, 1, 2 * sstep, SET, end / 2);         \
                                                              \
        for (i = 0; i < 16; i++)                              \
        {                                                     \
            assign(dst[i * dstep], e_32[i] + o_32[i]);        \
            assign(dst[(31 - i) * dstep], e_32[i] - o_32[i]); \
        }                                                     \
    } while (0)

#define IDCT_VAR4(H) \
    int limit2 = UHDMIN(col_limit + 4, H)
#define IDCT_VAR8(H)                  \
    int limit = UHDMIN(col_limit, H); \
    int limit2 = UHDMIN(col_limit + 4, H)
#define IDCT_VAR16(H) IDCT_VAR8(H)
#define IDCT_VAR32(H) IDCT_VAR8(H)

#define IDCT(H)                                         \
    static void FUNC(idct_##H##x##H)(                   \
        int16_t * coeffs, int col_limit)                \
    {                                                   \
        int i;                                          \
        int shift = 7;                                  \
        int add = 1 << (shift - 1);                     \
        int16_t *src = coeffs;                          \
        IDCT_VAR##H(H);                                 \
                                                        \
        for (i = 0; i < H; i++)                         \
        {                                               \
            TR_##H(src, src, H, H, SCALE, limit2);      \
            if (limit2 < H && i % 4 == 0 && !!i)        \
                limit2 -= 4;                            \
            src++;                                      \
        }                                               \
                                                        \
        shift = 20 - BIT_DEPTH;                         \
        add = 1 << (shift - 1);                         \
        for (i = 0; i < H; i++)                         \
        {                                               \
            TR_##H(coeffs, coeffs, 1, 1, SCALE, limit); \
            coeffs += H;                                \
        }                                               \
    }

#define IDCT_DC(H)                                           \
    static void FUNC(idct_##H##x##H##_dc)(                   \
        int16_t * coeffs)                                    \
    {                                                        \
        int i, j;                                            \
        int shift = 14 - BIT_DEPTH;                          \
        int add = 1 << (shift - 1);                          \
        int coeff = (((coeffs[0] + 1) >> 1) + add) >> shift; \
                                                             \
        for (j = 0; j < H; j++)                              \
        {                                                    \
            for (i = 0; i < H; i++)                          \
            {                                                \
                coeffs[i + j * H] = coeff;                   \
            }                                                \
        }                                                    \
    }

IDCT(4)
IDCT(8)
IDCT(16)
IDCT(32)

IDCT_DC(4)
IDCT_DC(8)
IDCT_DC(16)
IDCT_DC(32)

#undef TR_4
#undef TR_8
#undef TR_16
#undef TR_32

#undef SET
#undef SCALE
#undef ADD_AND_SCALE

static void FUNC(sao_band_filter)(uint8_t *_dst, uint8_t *_src,
                                  ptrdiff_t stride_dst, ptrdiff_t stride_src,
                                  int16_t *sao_offset_val, int sao_left_class,
                                  int width, int height)
{
    pixel *dst = (pixel *)_dst;
    pixel *src = (pixel *)_src;
    int offset_table[32] = {0};
    int k, y, x;
    int shift = BIT_DEPTH - 5;

    stride_dst /= sizeof(pixel);
    stride_src /= sizeof(pixel);

    for (k = 0; k < 4; k++)
    {
        offset_table[(k + sao_left_class) & 31] = sao_offset_val[k + 1];
    }
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(src[x] + offset_table[src[x] >> shift]);
        }
        dst += stride_dst;
        src += stride_src;
    }
}

#define VECTOR_SAO

#define CMP(a, b) (((a) > (b)) - ((a) < (b)))

static void FUNC(sao_edge_filter)(uint8_t *_dst, uint8_t *_src, ptrdiff_t stride_dst, int16_t *sao_offset_val,
                                  int eo, int width, int height)
{

    static const uint8_t edge_idx[] = {1, 2, 0, 3, 4};
    static const int8_t pos[4][2][2] =
        {
            {{-1, 0}, {1, 0}},  // horizontal
            {{0, -1}, {0, 1}},  // vertical
            {{-1, -1}, {1, 1}}, // 45 degree
            {{1, -1}, {-1, 1}}, // 135 degree
        };
    pixel *dst = (pixel *)_dst;
    pixel *src = (pixel *)_src;
    int a_stride, b_stride;
    int x, y;
    ptrdiff_t stride_src = (2 * MAX_PB_SIZE + UHD_INPUT_BUFFER_PADDING_SIZE) / sizeof(pixel);
    stride_dst /= sizeof(pixel);

    a_stride = pos[eo][0][0] + pos[eo][0][1] * (int)stride_src;
    b_stride = pos[eo][1][0] + pos[eo][1][1] * (int)stride_src;

#ifdef SCALAR_SAO
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            int diff0 = CMP(src[x], src[x + a_stride]);
            int diff1 = CMP(src[x], src[x + b_stride]);
            int offset_val = edge_idx[2 + diff0 + diff1];
            dst[x] = uhd_clip_pixel(src[x] + sao_offset_val[offset_val]);
        }
        src += stride_src;
        dst += stride_dst;
    }
#endif

#ifdef VECTOR_SAO
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x += 8)
        {
// for bit_depth > 8 the if block will be executed otherwise it will go to else part of src loading
#if BIT_DEPTH > 8
            uint16x8_t src0 = vld1q_u16(src + x);
            uint16x8_t src1 = vld1q_u16(src + x + a_stride);
            uint16x8_t src2 = vld1q_u16(src + x + b_stride);
#else
            uint16x8_t src0 = vmovl_u8(vld1_u8(src + x));
            uint16x8_t src1 = vmovl_u8(vld1_u8(src + x + a_stride));
            uint16x8_t src2 = vmovl_u8(vld1_u8(src + x + b_stride));
#endif

            // computes a>b - a<b to calculate diff0 and diff1
            uint16x8_t gt0 = vcgtq_u16(src0, src1);
            uint16x8_t lt0 = vcltq_u16(src0, src1);
            uint16x8_t gt1 = vcgtq_u16(src0, src2);
            uint16x8_t lt1 = vcltq_u16(src0, src2);

            int8x8_t diff0 = vmovn_s16(vreinterpretq_s16_u16(vsubq_u16(lt0, gt0)));
            int8x8_t diff1 = vmovn_s16(vreinterpretq_s16_u16(vsubq_u16(lt1, gt1)));

            // calculating offset value
            int8x8_t edge_index = vreinterpret_s8_u8(vld1_u8(edge_idx));
            int8x8_t ed_pos = vadd_s8(vadd_s8(vdup_n_s8(2), diff0), diff1);
            int8x8_t off_val = vtbl1_s8(edge_index, ed_pos);

            // calculates sao offset value
            int16x8_t sao = vld1q_s16(sao_offset_val);
            int8x8_t s_o = vqmovn_s16(sao);
            int16x8_t offset = vmovl_s8(vtbl1_s8(s_o, off_val));

            // adding src and sao offset value
            int32x4_t f_add0 = vuqaddq_s32(vmovl_s16(vget_low_s16(offset)), vmovl_u16(vget_low_u16(src0)));
            int32x4_t f_add1 = vuqaddq_s32(vmovl_s16(vget_high_s16(offset)), vmovl_u16(vget_high_u16(src0)));

            // clipping operation
            uint16x4_t clip0 = vqmovun_s32(f_add0);
            uint16x4_t clip1 = vqmovun_s32(f_add1);

            uint16x8_t opp = vcombine_u16(clip0, clip1);
            #if BIT_DEPTH > 8
                vst1q_u16(dst + x, opp);
            #else
            uint8x8_t finale = vqmovn_u16(opp);
            vst1_u8(dst + x, finale);
            #endif
        }
        src += stride_src;
        dst += stride_dst;
    }
#endif
}

static void FUNC(clip_row)(uint8_t *_dst, uint8_t *_src, int width, int offset_val)
{
    pixel *dst = (pixel *)_dst;
    pixel *src = (pixel *)_src;

    for (int x = 0; x < width; x++)
    {
        dst[x] = uhd_clip_pixel(src[x] + offset_val);
    }
}

static void FUNC(sao_edge_restore_0)(uint8_t *_dst, uint8_t *_src,
                                     ptrdiff_t stride_dst, ptrdiff_t stride_src, SAOParams *sao,
                                     int *borders, int _width, int _height,
                                     int c_idx, uint8_t *vert_edge,
                                     uint8_t *horiz_edge, uint8_t *diag_edge, func_clip_row_ptr clip_func)
{
    int x, y;
    pixel *dst = (pixel *)_dst;
    pixel *src = (pixel *)_src;
    int16_t *sao_offset_val = sao->offset_val[c_idx];
    int sao_eo_class = sao->eo_class[c_idx];
    int width = _width, height = _height;

    stride_dst /= sizeof(pixel);
    stride_src /= sizeof(pixel);

    if (sao_eo_class != SAO_EO_HORIZ)
    {
        if (borders[1])
        {
            clip_func(_dst, _src, width, sao_offset_val[0]);
        }
        if (borders[3])
        {
            int y_stride_dst = (int)stride_dst * (height - 1);
            int y_stride_src = (int)stride_src * (height - 1);

            clip_func((uint8_t *)(dst + y_stride_dst), (uint8_t *)(src + y_stride_src), width, sao_offset_val[0]);
            height--;
        }
    }
    if (sao_eo_class != SAO_EO_VERT)
    {
        if (borders[0])
        {
            int offset_val = sao_offset_val[0];
            for (y = 0; y < height; y++)
            {
                dst[y * stride_dst] = uhd_clip_pixel(src[y * stride_src] + offset_val);
            }
        }
        if (borders[2])
        {
            int offset_val = sao_offset_val[0];
            int offset = width - 1;
            for (x = 0; x < height; x++)
            {
                dst[x * stride_dst + offset] = uhd_clip_pixel(src[x * stride_src + offset] + offset_val);
            }
            width--;
        }
    }
}

static void FUNC(sao_edge_restore_1)(uint8_t *_dst, uint8_t *_src,
                                     ptrdiff_t stride_dst, ptrdiff_t stride_src, SAOParams *sao,
                                     int *borders, int _width, int _height,
                                     int c_idx, uint8_t *vert_edge,
                                     uint8_t *horiz_edge, uint8_t *diag_edge, func_clip_row_ptr clip_func)
{
    int x, y;
    pixel *dst = (pixel *)_dst;
    pixel *src = (pixel *)_src;
    int16_t *sao_offset_val = sao->offset_val[c_idx];
    int sao_eo_class = sao->eo_class[c_idx];
    int init_x = 0, init_y = 0, width = _width, height = _height;

    stride_dst /= sizeof(pixel);
    stride_src /= sizeof(pixel);

    if (sao_eo_class != SAO_EO_HORIZ)
    {
        if (borders[1])
        {
            clip_func(_dst, _src, width, sao_offset_val[0]);
            init_y = 1;
        }
        if (borders[3])
        {
            int y_stride_dst = (int)stride_dst * (height - 1);
            int y_stride_src = (int)stride_src * (height - 1);
            clip_func((uint8_t *)(dst + y_stride_dst), (uint8_t *)(src + y_stride_src), width, sao_offset_val[0]);
            height--;
        }
    }

    if (sao_eo_class != SAO_EO_VERT)
    {
        if (borders[0])
        {
            int offset_val = sao_offset_val[0];
            for (y = 0; y < height; y++)
            {
                dst[y * stride_dst] = uhd_clip_pixel(src[y * stride_src] + offset_val);
            }
            init_x = 1;
        }
        if (borders[2])
        {
            int offset_val = sao_offset_val[0];
            int offset = width - 1;
            for (x = 0; x < height; x++)
            {
                dst[x * stride_dst + offset] = uhd_clip_pixel(src[x * stride_src + offset] + offset_val);
            }
            width--;
        }
    }

    {
        int save_upper_left = !diag_edge[0] && sao_eo_class == SAO_EO_135D && !borders[0] && !borders[1];
        int save_upper_right = !diag_edge[1] && sao_eo_class == SAO_EO_45D && !borders[1] && !borders[2];
        int save_lower_right = !diag_edge[2] && sao_eo_class == SAO_EO_135D && !borders[2] && !borders[3];
        int save_lower_left = !diag_edge[3] && sao_eo_class == SAO_EO_45D && !borders[0] && !borders[3];

        // Restore pixels that can't be modified
        if (vert_edge[0] && sao_eo_class != SAO_EO_VERT)
        {
            for (y = init_y + save_upper_left; y < height - save_lower_left; y++)
            {
                dst[y * stride_dst] = src[y * stride_src];
            }
        }
        if (vert_edge[1] && sao_eo_class != SAO_EO_VERT)
        {
            for (y = init_y + save_upper_right; y < height - save_lower_right; y++)
            {
                dst[y * stride_dst + width - 1] = src[y * stride_src + width - 1];
            }
        }

        if (horiz_edge[0] && sao_eo_class != SAO_EO_HORIZ)
        {
            memcpy(dst + init_x + save_upper_left, src + init_x + save_upper_left,
                   (width - save_upper_right - init_x - save_upper_left) * sizeof(pixel));
        }
        if (horiz_edge[1] && sao_eo_class != SAO_EO_HORIZ)
        {
            memcpy(dst + (height - 1) * stride_dst + init_x + save_lower_left,
                   src + (height - 1) * stride_src + init_x + save_lower_left,
                   (width - save_lower_right - init_x - save_lower_left) * sizeof(pixel));
        }
        if (diag_edge[0] && sao_eo_class == SAO_EO_135D)
        {
            dst[0] = src[0];
        }
        if (diag_edge[1] && sao_eo_class == SAO_EO_45D)
        {
            dst[width - 1] = src[width - 1];
        }
        if (diag_edge[2] && sao_eo_class == SAO_EO_135D)
        {
            dst[stride_dst * (height - 1) + width - 1] = src[stride_src * (height - 1) + width - 1];
        }
        if (diag_edge[3] && sao_eo_class == SAO_EO_45D)
        {
            dst[stride_dst * (height - 1)] = src[stride_src * (height - 1)];
        }
    }
}

#undef CMP

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
static void FUNC(put_hevc_pel_pixels)(int16_t *dst,
                                      uint8_t *_src, ptrdiff_t _srcstride,
                                      int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = src[x] << (14 - BIT_DEPTH);
        }
        src += srcstride;
        dst += MAX_PB_SIZE;
    }
}

static void FUNC(put_hevc_pel_uni_pixels)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                          int height, intptr_t mx, intptr_t my, int width)
{
    int y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);

    for (y = 0; y < height; y++)
    {
        memcpy(dst, src, width * sizeof(pixel));
        src += srcstride;
        dst += dststride;
    }
}

static void FUNC(put_hevc_pel_bi_pixels)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                         int16_t *src2,
                                         int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);

    int shift = 14 + 1 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((src[x] << (14 - BIT_DEPTH)) + src2[x] + offset) >> shift);
        }
        src += srcstride;
        dst += dststride;
        src2 += MAX_PB_SIZE;
    }
}

static void FUNC(put_hevc_pel_uni_w_pixels)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                            int height, int denom, int wx, int ox, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    int shift = denom + 14 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    ox = ox * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel((((src[x] << (14 - BIT_DEPTH)) * wx + offset) >> shift) + ox);
        }
        src += srcstride;
        dst += dststride;
    }
}

static void FUNC(put_hevc_pel_bi_w_pixels)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                           int16_t *src2,
                                           int height, int denom, int wx0, int wx1,
                                           int ox0, int ox1, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);

    int shift = 14 + 1 - BIT_DEPTH;
    int log2Wd = denom + shift - 1;

    ox0 = ox0 * (1 << (BIT_DEPTH - 8));
    ox1 = ox1 * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((src[x] << (14 - BIT_DEPTH)) * wx1 + src2[x] * wx0 + ((ox0 + ox1 + 1) << log2Wd)) >>
                                    (log2Wd + 1));
        }
        src += srcstride;
        dst += dststride;
        src2 += MAX_PB_SIZE;
    }
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
#define QPEL_FILTER(src, stride)       \
    (filter[0] * src[x - 3 * stride] + \
     filter[1] * src[x - 2 * stride] + \
     filter[2] * src[x - stride] +     \
     filter[3] * src[x] +              \
     filter[4] * src[x + stride] +     \
     filter[5] * src[x + 2 * stride] + \
     filter[6] * src[x + 3 * stride] + \
     filter[7] * src[x + 4 * stride])

static void FUNC(put_hevc_qpel_h)(int16_t *dst,
                                  uint8_t *_src, ptrdiff_t _srcstride,
                                  int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_qpel_filters[mx - 1];
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = QPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        dst += MAX_PB_SIZE;
    }
}

static void FUNC(put_hevc_qpel_v)(int16_t *dst,
                                  uint8_t *_src, ptrdiff_t _srcstride,
                                  int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_qpel_filters[my - 1];
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = QPEL_FILTER(src, srcstride) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        dst += MAX_PB_SIZE;
    }
}

// START OF THE FUNCTION  -> QPEL_HV
#define VECTOR_LOGIC2_LOOP1
#define VECTOR_LOGIC2_LOOP2

static void FUNC(put_hevc_qpel_hv)(int16_t *dst,
                                   uint8_t *_src,
                                   ptrdiff_t _srcstride,
                                   int height, intptr_t mx,
                                   intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    int16_t tmp_array[(MAX_PB_SIZE + QPEL_EXTRA) * MAX_PB_SIZE];
    int16_t *tmp = tmp_array;
    src -= QPEL_EXTRA_BEFORE * srcstride;
    const int16_t *filter;
    filter = qpel_filter_size8[mx - 1];

#ifdef SCALAR_LOOP1
    for (y = 0; y < height + QPEL_EXTRA; y++)
    {
        for (x = 0; x < width; x++)
        {
            tmp[x] = QPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
#endif

#ifdef VECTOR_LOGIC2_LOOP1
    int shift = BIT_DEPTH - 8;
    int yval = height + QPEL_EXTRA;
    filter = qpel_filter_size8[mx - 1];

    #if BIT_DEPTH > 8
            int16x4_t fil0 = vdup_n_s16(filter[0]);
            int16x4_t fil1 = vdup_n_s16(filter[1]);
            int16x4_t fil2 = vdup_n_s16(filter[2]);
            int16x4_t fil3 = vdup_n_s16(filter[3]);
            int16x4_t fil4 = vdup_n_s16(filter[4]);
            int16x4_t fil5 = vdup_n_s16(filter[5]);
            int16x4_t fil6 = vdup_n_s16(filter[6]);
            int16x4_t fil7 = vdup_n_s16(filter[7]);
    #else
            int16x8_t fil0 = vdupq_n_s16(filter[0]);
            int16x8_t fil1 = vdupq_n_s16(filter[1]);
            int16x8_t fil2 = vdupq_n_s16(filter[2]);
            int16x8_t fil3 = vdupq_n_s16(filter[3]);
            int16x8_t fil4 = vdupq_n_s16(filter[4]);
            int16x8_t fil5 = vdupq_n_s16(filter[5]);
            int16x8_t fil6 = vdupq_n_s16(filter[6]);
            int16x8_t fil7 = vdupq_n_s16(filter[7]);
    #endif

    for (y = 0; y < yval; y++)
    {
        for (x = 0; x < width; x += 8)
        {
        #if BIT_DEPTH > 8
            int32x4_t final1 = vdupq_n_s32(0);
            int32x4_t final2 = vdupq_n_s32(0);

            uint16_t *ptr;
            ptr = src + x;
              int16x4_t src0 = vreinterpret_s16_u16(vld1_u16(ptr - 3));
            int16x4_t src1 = vreinterpret_s16_u16(vld1_u16(ptr + 1));

            int16x4_t src2 = vreinterpret_s16_u16(vld1_u16(ptr - 2));
            int16x4_t src3 = vreinterpret_s16_u16(vld1_u16(ptr + 2));

            int16x4_t src4 = vreinterpret_s16_u16(vld1_u16(ptr - 1));
            int16x4_t src5 = vreinterpret_s16_u16(vld1_u16(ptr + 3));

            int16x4_t src6 = vreinterpret_s16_u16(vld1_u16(ptr));
            int16x4_t src7 = vreinterpret_s16_u16(vld1_u16(ptr + 4));

            int16x4_t src8 = vreinterpret_s16_u16(vld1_u16(ptr + 1));
            int16x4_t src9 = vreinterpret_s16_u16(vld1_u16(ptr + 5));

            int16x4_t src10 = vreinterpret_s16_u16(vld1_u16(ptr + 2));
            int16x4_t src11 = vreinterpret_s16_u16(vld1_u16(ptr + 6));

            int16x4_t src12 = vreinterpret_s16_u16(vld1_u16(ptr + 3));
            int16x4_t src13 = vreinterpret_s16_u16(vld1_u16(ptr + 7));

            int16x4_t src14 = vreinterpret_s16_u16(vld1_u16(ptr + 4));
            int16x4_t src15 = vreinterpret_s16_u16(vld1_u16(ptr + 8));

            final1 = vmlal_s16(final1, src0 , fil0);
            final2 = vmlal_s16(final2, src1 , fil0);

            final1 = vmlal_s16(final1, src2 , fil1);
            final2 = vmlal_s16(final2, src3 , fil1);

            final1 = vmlal_s16(final1, src4 , fil2);
            final2 = vmlal_s16(final2, src5 , fil2);

            final1 = vmlal_s16(final1, src6 , fil3);
            final2 = vmlal_s16(final2, src7 , fil3);

            final1 = vmlal_s16(final1, src8 , fil4);
            final2 = vmlal_s16(final2, src9 , fil4);

            final1 = vmlal_s16(final1, src10 , fil5);
            final2 = vmlal_s16(final2, src11 , fil5);

            final1 = vmlal_s16(final1, src12 , fil6);
            final2 = vmlal_s16(final2, src13 , fil6);

            final1 = vmlal_s16(final1, src14 , fil7);
            final2 = vmlal_s16(final2, src15 , fil7);

            final1 = vshrq_n_s32(final1, shift);
            final2 = vshrq_n_s32(final2, shift);

            int16x4_t intt1 = vmovn_s32(final1);
            int16x4_t intt2 = vmovn_s32(final2);

            vst1_s16(tmp + x, intt1);
            vst1_s16(tmp + x + 4, intt2);

        #else
            int16x8_t final = vdupq_n_s16(0);
            uint8_t *ptr;
            ptr = src + x;
            int16x8_t src0 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr - 3)));
            int16x8_t src1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr - 2)));
            int16x8_t src2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr - 1)));
            int16x8_t src3 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr)));
            int16x8_t src4 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr + 1)));
            int16x8_t src5 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr + 2)));
            int16x8_t src6 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr + 3)));
            int16x8_t src7 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr + 4)));

            final = vmlaq_s16(final, src0, fil0);
            final = vmlaq_s16(final, src1, fil1);
            final = vmlaq_s16(final, src2, fil2);
            final = vmlaq_s16(final, src3, fil3);
            final = vmlaq_s16(final, src4, fil4);
            final = vmlaq_s16(final, src5, fil5);
            final = vmlaq_s16(final, src6, fil6);
            final = vmlaq_s16(final, src7, fil7);
            final = vshrq_n_s16(final, shift);
            vst1q_s16(tmp + x, final);
    #endif
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
#endif

#ifdef SCALAR_LOOP2
    tmp = tmp_array + QPEL_EXTRA_BEFORE * MAX_PB_SIZE;
    filter = qpel_filter_size8[my - 1];
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = QPEL_FILTER(tmp, MAX_PB_SIZE) >> 6;
        }
        tmp += MAX_PB_SIZE;
        dst += MAX_PB_SIZE;
    }
#endif

#ifdef VECTOR_LOGIC2_LOOP2
    int16_t *str;
    tmp = tmp_array + QPEL_EXTRA_BEFORE * MAX_PB_SIZE;
    filter = qpel_filter_size8[my - 1];

    int16x4_t filt0 = vdup_n_s16(filter[0]);
    int16x4_t filt1 = vdup_n_s16(filter[1]);
    int16x4_t filt2 = vdup_n_s16(filter[2]);
    int16x4_t filt3 = vdup_n_s16(filter[3]);
    int16x4_t filt4 = vdup_n_s16(filter[4]);
    int16x4_t filt5 = vdup_n_s16(filter[5]);
    int16x4_t filt6 = vdup_n_s16(filter[6]);
    int16x4_t filt7 = vdup_n_s16(filter[7]);

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x += 8)
        {
            int32x4_t final1 = vdupq_n_s32(0);
            int32x4_t final2 = vdupq_n_s32(0);

            str = tmp + x;
            int16x4_t src0 = vld1_s16(str - 192);
            int16x4_t src1 = vld1_s16(str - 188);

            int16x4_t src2 = vld1_s16(str - 128);
            int16x4_t src3 = vld1_s16(str - 124);

            int16x4_t src4 = vld1_s16(str - 64);
            int16x4_t src5 = vld1_s16(str - 60);

            int16x4_t src6 = vld1_s16(str);
            int16x4_t src7 = vld1_s16(str + 4);

            int16x4_t src8 = vld1_s16(str + 64);
            int16x4_t src9 = vld1_s16(str + 68);

            int16x4_t src10 = vld1_s16(str + 128);
            int16x4_t src11 = vld1_s16(str + 132);

            int16x4_t src12 = vld1_s16(str + 192);
            int16x4_t src13 = vld1_s16(str + 196);

            int16x4_t src14 = vld1_s16(str + 256);
            int16x4_t src15 = vld1_s16(str + 260);

            final1 = vmlal_s16(final1, src0, filt0);
            final2 = vmlal_s16(final2, src1, filt0);

            final1 = vmlal_s16(final1, src2, filt1);
            final2 = vmlal_s16(final2, src3, filt1);

            final1 = vmlal_s16(final1, src4, filt2);
            final2 = vmlal_s16(final2, src5, filt2);

            final1 = vmlal_s16(final1, src6, filt3);
            final2 = vmlal_s16(final2, src7, filt3);

            final1 = vmlal_s16(final1, src8, filt4);
            final2 = vmlal_s16(final2, src9, filt4);

            final1 = vmlal_s16(final1, src10, filt5);
            final2 = vmlal_s16(final2, src11, filt5);

            final1 = vmlal_s16(final1, src12, filt6);
            final2 = vmlal_s16(final2, src13, filt6);

            final1 = vmlal_s16(final1, src14, filt7);
            final2 = vmlal_s16(final2, src15, filt7);

            final1 = vshrq_n_s32(final1, 6);
            final2 = vshrq_n_s32(final2, 6);

            int16x4_t f_sum0 = vqmovn_s32(final1);
            int16x4_t f_sum1 = vqmovn_s32(final2);

            vst1_s16(dst + x, f_sum0);
            vst1_s16(dst + x + 4, f_sum1);
        }
        tmp += MAX_PB_SIZE;
        dst += MAX_PB_SIZE;
    }
#endif
}
// END OF THE FUNCTION

static void FUNC(put_hevc_qpel_uni_h)(uint8_t *_dst, ptrdiff_t _dststride,
                                      uint8_t *_src, ptrdiff_t _srcstride,
                                      int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_qpel_filters[mx - 1];
    int shift = 14 - BIT_DEPTH;

#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((QPEL_FILTER(src, 1) >> (BIT_DEPTH - 8)) + offset) >> shift);
        }
        src += srcstride;
        dst += dststride;
    }
}

static void FUNC(put_hevc_qpel_bi_h)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                     int16_t *src2,
                                     int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);

    const int8_t *filter = uhd_hevc_qpel_filters[mx - 1];

    int shift = 14 + 1 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((QPEL_FILTER(src, 1) >> (BIT_DEPTH - 8)) + src2[x] + offset) >> shift);
        }
        src += srcstride;
        dst += dststride;
        src2 += MAX_PB_SIZE;
    }
}

static void FUNC(put_hevc_qpel_uni_v)(uint8_t *_dst, ptrdiff_t _dststride,
                                      uint8_t *_src, ptrdiff_t _srcstride,
                                      int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_qpel_filters[my - 1];
    int shift = 14 - BIT_DEPTH;

#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((QPEL_FILTER(src, srcstride) >> (BIT_DEPTH - 8)) + offset) >> shift);
        }
        src += srcstride;
        dst += dststride;
    }
}

static void FUNC(put_hevc_qpel_bi_v)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                     int16_t *src2,
                                     int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);

    const int8_t *filter = uhd_hevc_qpel_filters[my - 1];

    int shift = 14 + 1 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((QPEL_FILTER(src, srcstride) >> (BIT_DEPTH - 8)) + src2[x] + offset) >> shift);
        }
        src += srcstride;
        dst += dststride;
        src2 += MAX_PB_SIZE;
    }
}

// START OF QPEL_UNI_HV

#define QPEL_UNI_VECTOR1
#define QPEL_UNI_VECTOR2

static void FUNC(put_hevc_qpel_uni_hv)(uint8_t *_dst, ptrdiff_t _dststride,
                                       uint8_t *_src, ptrdiff_t _srcstride,
                                       int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    const int16_t *filter;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    int16_t tmp_array[(MAX_PB_SIZE + QPEL_EXTRA) * MAX_PB_SIZE];
    int16_t *tmp = tmp_array;
    int shift = 14 - BIT_DEPTH;

#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    src -= QPEL_EXTRA_BEFORE * srcstride;

#ifdef QPEL_UNI_SCALAR1
    filter = qpel_filter_size8[mx - 1];
    for (y = 0; y < height + QPEL_EXTRA; y++)
    {
        for (x = 0; x < width; x++)
        {
            tmp[x] = QPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
#endif

#ifdef QPEL_UNI_VECTOR1
    int shifts = BIT_DEPTH - 8;
    int yval = height + QPEL_EXTRA;
    filter = qpel_filter_size8[mx - 1];

    #if BIT_DEPTH > 8
            int16x4_t fil0 = vdup_n_s16(filter[0]);
            int16x4_t fil1 = vdup_n_s16(filter[1]);
            int16x4_t fil2 = vdup_n_s16(filter[2]);
            int16x4_t fil3 = vdup_n_s16(filter[3]);
            int16x4_t fil4 = vdup_n_s16(filter[4]);
            int16x4_t fil5 = vdup_n_s16(filter[5]);
            int16x4_t fil6 = vdup_n_s16(filter[6]);
            int16x4_t fil7 = vdup_n_s16(filter[7]);
    #else
            int16x8_t fil0 = vdupq_n_s16(filter[0]);
            int16x8_t fil1 = vdupq_n_s16(filter[1]);
            int16x8_t fil2 = vdupq_n_s16(filter[2]);
            int16x8_t fil3 = vdupq_n_s16(filter[3]);
            int16x8_t fil4 = vdupq_n_s16(filter[4]);
            int16x8_t fil5 = vdupq_n_s16(filter[5]);
            int16x8_t fil6 = vdupq_n_s16(filter[6]);
            int16x8_t fil7 = vdupq_n_s16(filter[7]);
    #endif

    for (y = 0; y < yval; y++)
    {
        for (x = 0; x < width; x += 8)
        {
        #if BIT_DEPTH > 8
            int32x4_t final1 = vdupq_n_s32(0);
            int32x4_t final2 = vdupq_n_s32(0);

            uint16_t *ptr;
            ptr = src + x;
            int16x4_t src0 = vreinterpret_s16_u16(vld1_u16(ptr - 3));
            int16x4_t src1 = vreinterpret_s16_u16(vld1_u16(ptr + 1));

            int16x4_t src2 = vreinterpret_s16_u16(vld1_u16(ptr - 2));
            int16x4_t src3 = vreinterpret_s16_u16(vld1_u16(ptr + 2));

            int16x4_t src4 = vreinterpret_s16_u16(vld1_u16(ptr - 1));
            int16x4_t src5 = vreinterpret_s16_u16(vld1_u16(ptr + 3));

            int16x4_t src6 = vreinterpret_s16_u16(vld1_u16(ptr));
            int16x4_t src7 = vreinterpret_s16_u16(vld1_u16(ptr + 4));

            int16x4_t src8 = vreinterpret_s16_u16(vld1_u16(ptr + 1));
            int16x4_t src9 = vreinterpret_s16_u16(vld1_u16(ptr + 5));

            int16x4_t src10 = vreinterpret_s16_u16(vld1_u16(ptr + 2));
            int16x4_t src11 = vreinterpret_s16_u16(vld1_u16(ptr + 6));

            int16x4_t src12 = vreinterpret_s16_u16(vld1_u16(ptr + 3));
            int16x4_t src13 = vreinterpret_s16_u16(vld1_u16(ptr + 7));

            int16x4_t src14 = vreinterpret_s16_u16(vld1_u16(ptr + 4));
            int16x4_t src15 = vreinterpret_s16_u16(vld1_u16(ptr + 8));


            final1 = vmlal_s16(final1, src0 , fil0);
            final2 = vmlal_s16(final2, src1 , fil0);

            final1 = vmlal_s16(final1, src2 , fil1);
            final2 = vmlal_s16(final2, src3 , fil1);

            final1 = vmlal_s16(final1, src4 , fil2);
            final2 = vmlal_s16(final2, src5 , fil2);

            final1 = vmlal_s16(final1, src6 , fil3);
            final2 = vmlal_s16(final2, src7 , fil3);

            final1 = vmlal_s16(final1, src8 , fil4);
            final2 = vmlal_s16(final2, src9 , fil4);

            final1 = vmlal_s16(final1, src10 , fil5);
            final2 = vmlal_s16(final2, src11 , fil5);

            final1 = vmlal_s16(final1, src12 , fil6);
            final2 = vmlal_s16(final2, src13 , fil6);

            final1 = vmlal_s16(final1, src14 , fil7);
            final2 = vmlal_s16(final2, src15 , fil7);

            final1 = vshrq_n_s32(final1, shifts);
            final2 = vshrq_n_s32(final2, shifts);

            int16x4_t intt1 = vmovn_s32(final1);
            int16x4_t intt2 = vmovn_s32(final2);

            vst1_s16(tmp + x, intt1);
            vst1_s16(tmp + x + 4, intt2);

        #else
            int16x8_t final = vdupq_n_s16(0);
            uint8_t *ptr;
            ptr = src + x;
            int16x8_t src0 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr - 3)));
            int16x8_t src1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr - 2)));
            int16x8_t src2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr - 1)));
            int16x8_t src3 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr)));
            int16x8_t src4 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr + 1)));
            int16x8_t src5 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr + 2)));
            int16x8_t src6 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr + 3)));
            int16x8_t src7 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr + 4)));

            final = vmlaq_s16(final, src0, fil0);
            final = vmlaq_s16(final, src1, fil1);
            final = vmlaq_s16(final, src2, fil2);
            final = vmlaq_s16(final, src3, fil3);
            final = vmlaq_s16(final, src4, fil4);
            final = vmlaq_s16(final, src5, fil5);
            final = vmlaq_s16(final, src6, fil6);
            final = vmlaq_s16(final, src7, fil7);
            final = vshrq_n_s16(final, shifts);
            vst1q_s16(tmp + x, final);
    #endif
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
#endif

#ifdef QPEL_UNI_SCALAR2
    tmp = tmp_array + QPEL_EXTRA_BEFORE * MAX_PB_SIZE;
    filter = uhd_hevc_qpel_filters[my - 1];

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((QPEL_FILTER(tmp, MAX_PB_SIZE) >> 6) + offset) >> shift);
           // printf("\n%d",dst[x]);
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
    //exit(0);
#endif

#ifdef QPEL_UNI_VECTOR2
    int16_t *str;
    tmp = tmp_array + 192;
    filter = qpel_filter_size8[my - 1];

    int16x8_t offvector = vdupq_n_s16(offset);
    // int32x4_t max = vdupq_n_s32((1 << BIT_DEPTH) - 1);
    // int32x4_t min = vdupq_n_s32(0);

    int16x8_t max = vdupq_n_s16((1 << BIT_DEPTH) - 1);
    int16x8_t min = vdupq_n_s16(0);

    int16x4_t filt0 = vdup_n_s16(filter[0]);
    int16x4_t filt1 = vdup_n_s16(filter[1]);
    int16x4_t filt2 = vdup_n_s16(filter[2]);
    int16x4_t filt3 = vdup_n_s16(filter[3]);
    int16x4_t filt4 = vdup_n_s16(filter[4]);
    int16x4_t filt5 = vdup_n_s16(filter[5]);
    int16x4_t filt6 = vdup_n_s16(filter[6]);
    int16x4_t filt7 = vdup_n_s16(filter[7]);

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x += 8)
        {
            str = tmp + x;
            int32x4_t final1 = vdupq_n_s32(0);
            int32x4_t final2 = vdupq_n_s32(0);

            int16x4_t src0 = vld1_s16(str - 192);
            int16x4_t src1 = vld1_s16(str - 188);

            int16x4_t src2 = vld1_s16(str - 128);
            int16x4_t src3 = vld1_s16(str - 124);

            int16x4_t src4 = vld1_s16(str - 64);
            int16x4_t src5 = vld1_s16(str - 60);

            int16x4_t src6 = vld1_s16(str);
            int16x4_t src7 = vld1_s16(str + 4);

            int16x4_t src8 = vld1_s16(str + 64);
            int16x4_t src9 = vld1_s16(str + 68);

            int16x4_t src10 = vld1_s16(str + 128);
            int16x4_t src11 = vld1_s16(str + 132);

            int16x4_t src12 = vld1_s16(str + 192);
            int16x4_t src13 = vld1_s16(str + 196);

            int16x4_t src14 = vld1_s16(str + 256);
            int16x4_t src15 = vld1_s16(str + 260);

            final1 = vmlal_s16(final1, src0, filt0);
            final2 = vmlal_s16(final2, src1, filt0);

            final1 = vmlal_s16(final1, src2, filt1);
            final2 = vmlal_s16(final2, src3, filt1);

            final1 = vmlal_s16(final1, src4, filt2);
            final2 = vmlal_s16(final2, src5, filt2);

            final1 = vmlal_s16(final1, src6, filt3);
            final2 = vmlal_s16(final2, src7, filt3);

            final1 = vmlal_s16(final1, src8, filt4);
            final2 = vmlal_s16(final2, src9, filt4);

            final1 = vmlal_s16(final1, src10, filt5);
            final2 = vmlal_s16(final2, src11, filt5);

            final1 = vmlal_s16(final1, src12, filt6);
            final2 = vmlal_s16(final2, src13, filt6);

            final1 = vmlal_s16(final1, src14, filt7);
            final2 = vmlal_s16(final2, src15, filt7);

            int16x4_t f_sum0 = vshrn_n_s32(final1, 6);
            int16x4_t f_sum1 = vshrn_n_s32(final2, 6);

            int16x8_t combined = vcombine_s16(f_sum0, f_sum1);

            // int16x8_t srcvector = vld1q_s16(src2 + x);

            int16x8_t result = vaddq_s16(combined, offvector);
            result = vshrq_n_s16(result,shift);

            // int32x4_t result2 = vaddl_s16(vget_low_s16(result1), vget_low_s16(srcvector));
            // int32x4_t result3 = vaddl_s16(vget_high_s16(result1), vget_high_s16(srcvector));

            // result2 = vshrq_n_s32(result2, shift);
            // result3 = vshrq_n_s32(result3, shift);
            uint16x8_t clip00 = vreinterpretq_u16_s16((vminq_s16(max, vmaxq_s16(min, result))));


            // uint16x4_t clip0 = vreinterpret_u16_s16(vmovn_s32(vminq_s32(max, vmaxq_s32(min, result2))));
            // uint16x4_t clip1 = vreinterpret_u16_s16(vmovn_s32(vminq_s32(max, vmaxq_s32(min, result3))));

           // uint16x8_t opp = vcombine_u16(clip0, clip1);
            #if BIT_DEPTH > 8
                // for(int val=0;val<8;val++)
                // {
                //     printf("\n%d",clip00[val]);
                // }
                vst1q_u16(dst + x, clip00);
            #else
            uint8x8_t finale = vqmovn_u16(clip00);
                // for(int val=0;val<8;val++)
                // {
                //     printf("\n%d",finale[val]);
                // }
            vst1_u8(dst + x, finale);
            #endif
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
       // src2 += MAX_PB_SIZE;
    }
    // exit(0);
#endif
}

// START OF THE FUNCTION - QPEL_BI_HV
#define BI_VECTOR1
#define BI_VECTOR2

static void FUNC(put_hevc_qpel_bi_hv)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                      int16_t *src2,
                                      int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    const int16_t *filter;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst; 
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    int16_t tmp_array[(MAX_PB_SIZE + QPEL_EXTRA) * MAX_PB_SIZE];
    int16_t *tmp = tmp_array;

    int shift = 14 + 1 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    src -= QPEL_EXTRA_BEFORE * srcstride;

#ifdef BI_SCALAR1
    filter = qpel_filter_size8[mx - 1];
    for (y = 0; y < height + QPEL_EXTRA; y++)
    {
        for (x = 0; x < width; x++)
        {
            tmp[x] = QPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
#endif

#ifdef BI_VECTOR1
    int shifts = BIT_DEPTH - 8;
    int yval = height + QPEL_EXTRA;
    filter = qpel_filter_size8[mx - 1];

    #if BIT_DEPTH > 8
            int16x4_t fil0 = vdup_n_s16(filter[0]);
            int16x4_t fil1 = vdup_n_s16(filter[1]);
            int16x4_t fil2 = vdup_n_s16(filter[2]);
            int16x4_t fil3 = vdup_n_s16(filter[3]);
            int16x4_t fil4 = vdup_n_s16(filter[4]);
            int16x4_t fil5 = vdup_n_s16(filter[5]);
            int16x4_t fil6 = vdup_n_s16(filter[6]);
            int16x4_t fil7 = vdup_n_s16(filter[7]);
    #else
            int16x8_t fil0 = vdupq_n_s16(filter[0]);
            int16x8_t fil1 = vdupq_n_s16(filter[1]);
            int16x8_t fil2 = vdupq_n_s16(filter[2]);
            int16x8_t fil3 = vdupq_n_s16(filter[3]);
            int16x8_t fil4 = vdupq_n_s16(filter[4]);
            int16x8_t fil5 = vdupq_n_s16(filter[5]);
            int16x8_t fil6 = vdupq_n_s16(filter[6]);
            int16x8_t fil7 = vdupq_n_s16(filter[7]);
    #endif

    for (y = 0; y < yval; y++)
    {
        for (x = 0; x < width; x += 8)
        {
           

        #if BIT_DEPTH > 8
            int32x4_t final1 = vdupq_n_s32(0);
            int32x4_t final2 = vdupq_n_s32(0);

            uint16_t *ptr;
            ptr = src + x;
            int16x4_t src0 = vreinterpret_s16_u16(vld1_u16(ptr - 3));
            int16x4_t src1 = vreinterpret_s16_u16(vld1_u16(ptr + 1));

            int16x4_t Src2 = vreinterpret_s16_u16(vld1_u16(ptr - 2));
            int16x4_t src3 = vreinterpret_s16_u16(vld1_u16(ptr + 2));

            int16x4_t src4 = vreinterpret_s16_u16(vld1_u16(ptr - 1));
            int16x4_t src5 = vreinterpret_s16_u16(vld1_u16(ptr + 3));

            int16x4_t src6 = vreinterpret_s16_u16(vld1_u16(ptr));
            int16x4_t src7 = vreinterpret_s16_u16(vld1_u16(ptr + 4));

            int16x4_t src8 = vreinterpret_s16_u16(vld1_u16(ptr + 1));
            int16x4_t src9 = vreinterpret_s16_u16(vld1_u16(ptr + 5));

            int16x4_t src10 = vreinterpret_s16_u16(vld1_u16(ptr + 2));
            int16x4_t src11 = vreinterpret_s16_u16(vld1_u16(ptr + 6));

            int16x4_t src12 = vreinterpret_s16_u16(vld1_u16(ptr + 3));
            int16x4_t src13 = vreinterpret_s16_u16(vld1_u16(ptr + 7));

            int16x4_t src14 = vreinterpret_s16_u16(vld1_u16(ptr + 4));
            int16x4_t src15 = vreinterpret_s16_u16(vld1_u16(ptr + 8));


            final1 = vmlal_s16(final1, src0 , fil0);
            final2 = vmlal_s16(final2, src1 , fil0);

            final1 = vmlal_s16(final1, Src2 , fil1);
            final2 = vmlal_s16(final2, src3 , fil1);

            final1 = vmlal_s16(final1, src4 , fil2);
            final2 = vmlal_s16(final2, src5 , fil2);

            final1 = vmlal_s16(final1, src6 , fil3);
            final2 = vmlal_s16(final2, src7 , fil3);

            final1 = vmlal_s16(final1, src8 , fil4);
            final2 = vmlal_s16(final2, src9 , fil4);

            final1 = vmlal_s16(final1, src10 , fil5);
            final2 = vmlal_s16(final2, src11 , fil5);

            final1 = vmlal_s16(final1, src12 , fil6);
            final2 = vmlal_s16(final2, src13 , fil6);

            final1 = vmlal_s16(final1, src14 , fil7);
            final2 = vmlal_s16(final2, src15 , fil7);

            final1 = vshrq_n_s32(final1, shifts);
            final2 = vshrq_n_s32(final2, shifts);

            int16x4_t intt1 = vmovn_s32(final1);
            int16x4_t intt2 = vmovn_s32(final2);

            vst1_s16(tmp + x, intt1);
            vst1_s16(tmp + x + 4, intt2);

        #else
            int16x8_t final = vdupq_n_s16(0);
            uint8_t *ptr;
            ptr = src + x;
            int16x8_t src0 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr - 3)));
            int16x8_t src1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr - 2)));
            int16x8_t Src2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr - 1)));
            int16x8_t src3 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr)));
            int16x8_t src4 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr + 1)));
            int16x8_t src5 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr + 2)));
            int16x8_t src6 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr + 3)));
            int16x8_t src7 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr + 4)));

            final = vmlaq_s16(final, src0, fil0);
            final = vmlaq_s16(final, src1, fil1);
            final = vmlaq_s16(final, Src2, fil2);
            final = vmlaq_s16(final, src3, fil3);
            final = vmlaq_s16(final, src4, fil4);
            final = vmlaq_s16(final, src5, fil5);
            final = vmlaq_s16(final, src6, fil6);
            final = vmlaq_s16(final, src7, fil7);
            final = vshrq_n_s16(final, shift);
            vst1q_s16(tmp + x, final);
    #endif
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
#endif

#ifdef BI_SCALAR2
    tmp = tmp_array + QPEL_EXTRA_BEFORE * MAX_PB_SIZE;
    filter = uhd_hevc_qpel_filters[my - 1];
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((QPEL_FILTER(tmp, MAX_PB_SIZE) >> 6) + src2[x] + offset) >> shift);
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
        src2 += MAX_PB_SIZE;
    }
#endif

#ifdef BI_VECTOR2
    int16_t *str;
    tmp = tmp_array + 192;
    filter = qpel_filter_size8[my - 1];

    int16x8_t offvector = vdupq_n_s16(offset);
    int32x4_t max = vdupq_n_s32((1 << BIT_DEPTH) - 1);
    int32x4_t min = vdupq_n_s32(0);

    int16x4_t filt0 = vdup_n_s16(filter[0]);
    int16x4_t filt1 = vdup_n_s16(filter[1]);
    int16x4_t filt2 = vdup_n_s16(filter[2]);
    int16x4_t filt3 = vdup_n_s16(filter[3]);
    int16x4_t filt4 = vdup_n_s16(filter[4]);
    int16x4_t filt5 = vdup_n_s16(filter[5]);
    int16x4_t filt6 = vdup_n_s16(filter[6]);
    int16x4_t filt7 = vdup_n_s16(filter[7]);

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x += 8)
        {
            str = tmp + x;
            int32x4_t final1 = vdupq_n_s32(0);
            int32x4_t final2 = vdupq_n_s32(0);

            int16x4_t src0 = vld1_s16(str - 192);
            int16x4_t src1 = vld1_s16(str - 188);

            int16x4_t Src2 = vld1_s16(str - 128);
            int16x4_t src3 = vld1_s16(str - 124);

            int16x4_t src4 = vld1_s16(str - 64);
            int16x4_t src5 = vld1_s16(str - 60);

            int16x4_t src6 = vld1_s16(str);
            int16x4_t src7 = vld1_s16(str + 4);

            int16x4_t src8 = vld1_s16(str + 64);
            int16x4_t src9 = vld1_s16(str + 68);

            int16x4_t src10 = vld1_s16(str + 128);
            int16x4_t src11 = vld1_s16(str + 132);

            int16x4_t src12 = vld1_s16(str + 192);
            int16x4_t src13 = vld1_s16(str + 196);

            int16x4_t src14 = vld1_s16(str + 256);
            int16x4_t src15 = vld1_s16(str + 260);

            final1 = vmlal_s16(final1, src0, filt0);
            final2 = vmlal_s16(final2, src1, filt0);

            final1 = vmlal_s16(final1, Src2, filt1);
            final2 = vmlal_s16(final2, src3, filt1);

            final1 = vmlal_s16(final1, src4, filt2);
            final2 = vmlal_s16(final2, src5, filt2);

            final1 = vmlal_s16(final1, src6, filt3);
            final2 = vmlal_s16(final2, src7, filt3);

            final1 = vmlal_s16(final1, src8, filt4);
            final2 = vmlal_s16(final2, src9, filt4);

            final1 = vmlal_s16(final1, src10, filt5);
            final2 = vmlal_s16(final2, src11, filt5);

            final1 = vmlal_s16(final1, src12, filt6);
            final2 = vmlal_s16(final2, src13, filt6);

            final1 = vmlal_s16(final1, src14, filt7);
            final2 = vmlal_s16(final2, src15, filt7);

            int16x4_t f_sum0 = vshrn_n_s32(final1, 6);
            int16x4_t f_sum1 = vshrn_n_s32(final2, 6);

            int16x8_t combined = vcombine_s16(f_sum0, f_sum1);

            int16x8_t srcvector = vld1q_s16(src2 + x);

            int16x8_t result1 = vaddq_s16(combined, offvector);

            int32x4_t result2 = vaddl_s16(vget_low_s16(result1), vget_low_s16(srcvector));
            int32x4_t result3 = vaddl_s16(vget_high_s16(result1), vget_high_s16(srcvector));

            result2 = vshrq_n_s32(result2, shift);
            result3 = vshrq_n_s32(result3, shift);

            uint16x4_t clip0 = vreinterpret_u16_s16(vmovn_s32(vminq_s32(max, vmaxq_s32(min, result2))));
            uint16x4_t clip1 = vreinterpret_u16_s16(vmovn_s32(vminq_s32(max, vmaxq_s32(min, result3))));

            uint16x8_t opp = vcombine_u16(clip0, clip1);
            #if BIT_DEPTH > 8
                vst1q_u16(dst + x, opp);
            #else
            uint8x8_t finale = vqmovn_u16(opp);
            vst1_u8(dst + x, finale);
            #endif
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
        src2 += MAX_PB_SIZE;
    }
#endif
}
// END OF QPEL_BI_HV

static void FUNC(put_hevc_qpel_uni_w_h)(uint8_t *_dst, ptrdiff_t _dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride,
                                        int height, int denom, int wx, int ox,
                                        intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_qpel_filters[mx - 1];
    int shift = denom + 14 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    ox = ox * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel((((QPEL_FILTER(src, 1) >> (BIT_DEPTH - 8)) * wx + offset) >> shift) + ox);
        }
        src += srcstride;
        dst += dststride;
    }
}

static void FUNC(put_hevc_qpel_bi_w_h)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                       int16_t *src2,
                                       int height, int denom, int wx0, int wx1,
                                       int ox0, int ox1, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);

    const int8_t *filter = uhd_hevc_qpel_filters[mx - 1];

    int shift = 14 + 1 - BIT_DEPTH;
    int log2Wd = denom + shift - 1;

    ox0 = ox0 * (1 << (BIT_DEPTH - 8));
    ox1 = ox1 * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
            dst[x] = uhd_clip_pixel(((QPEL_FILTER(src, 1) >> (BIT_DEPTH - 8)) * wx1 + src2[x] * wx0 +
                                     ((ox0 + ox1 + 1) << log2Wd)) >>
                                    (log2Wd + 1));
        src += srcstride;
        dst += dststride;
        src2 += MAX_PB_SIZE;
    }
}

static void FUNC(put_hevc_qpel_uni_w_v)(uint8_t *_dst, ptrdiff_t _dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride,
                                        int height, int denom, int wx, int ox,
                                        intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_qpel_filters[my - 1];
    int shift = denom + 14 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    ox = ox * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel((((QPEL_FILTER(src, srcstride) >> (BIT_DEPTH - 8)) * wx + offset) >> shift) + ox);
        }
        src += srcstride;
        dst += dststride;
    }
}

static void FUNC(put_hevc_qpel_bi_w_v)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                       int16_t *src2,
                                       int height, int denom, int wx0, int wx1,
                                       int ox0, int ox1, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);

    const int8_t *filter = uhd_hevc_qpel_filters[my - 1];

    int shift = 14 + 1 - BIT_DEPTH;
    int log2Wd = denom + shift - 1;

    ox0 = ox0 * (1 << (BIT_DEPTH - 8));
    ox1 = ox1 * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
            dst[x] = uhd_clip_pixel(((QPEL_FILTER(src, srcstride) >> (BIT_DEPTH - 8)) * wx1 + src2[x] * wx0 +
                                     ((ox0 + ox1 + 1) << log2Wd)) >>
                                    (log2Wd + 1));
        src += srcstride;
        dst += dststride;
        src2 += MAX_PB_SIZE;
    }
}

static void FUNC(put_hevc_qpel_uni_w_hv)(uint8_t *_dst, ptrdiff_t _dststride,
                                         uint8_t *_src, ptrdiff_t _srcstride,
                                         int height, int denom, int wx, int ox,
                                         intptr_t mx, intptr_t my, int width)
{
    int x, y;
    const int8_t *filter;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    int16_t tmp_array[(MAX_PB_SIZE + QPEL_EXTRA) * MAX_PB_SIZE];
    int16_t *tmp = tmp_array;
    int shift = denom + 14 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    src -= QPEL_EXTRA_BEFORE * srcstride;
    filter = uhd_hevc_qpel_filters[mx - 1];
    for (y = 0; y < height + QPEL_EXTRA; y++)
    {
        for (x = 0; x < width; x++)
        {
            tmp[x] = QPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }

    tmp = tmp_array + QPEL_EXTRA_BEFORE * MAX_PB_SIZE;
    filter = uhd_hevc_qpel_filters[my - 1];

    ox = ox * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel((((QPEL_FILTER(tmp, MAX_PB_SIZE) >> 6) * wx + offset) >> shift) + ox);
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

static void FUNC(put_hevc_qpel_bi_w_hv)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                        int16_t *src2,
                                        int height, int denom, int wx0, int wx1,
                                        int ox0, int ox1, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    const int8_t *filter;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    int16_t tmp_array[(MAX_PB_SIZE + QPEL_EXTRA) * MAX_PB_SIZE];
    int16_t *tmp = tmp_array;
    int shift = 14 + 1 - BIT_DEPTH;
    int log2Wd = denom + shift - 1;

    src -= QPEL_EXTRA_BEFORE * srcstride;
    filter = uhd_hevc_qpel_filters[mx - 1];
    for (y = 0; y < height + QPEL_EXTRA; y++)
    {
        for (x = 0; x < width; x++)
        {
            tmp[x] = QPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }

    tmp = tmp_array + QPEL_EXTRA_BEFORE * MAX_PB_SIZE;
    filter = uhd_hevc_qpel_filters[my - 1];

    ox0 = ox0 * (1 << (BIT_DEPTH - 8));
    ox1 = ox1 * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
            dst[x] = uhd_clip_pixel(((QPEL_FILTER(tmp, MAX_PB_SIZE) >> 6) * wx1 + src2[x] * wx0 +
                                     ((ox0 + ox1 + 1) << log2Wd)) >>
                                    (log2Wd + 1));
        tmp += MAX_PB_SIZE;
        dst += dststride;
        src2 += MAX_PB_SIZE;
    }
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
#define EPEL_FILTER(src, stride)   \
    (filter[0] * src[x - stride] + \
     filter[1] * src[x] +          \
     filter[2] * src[x + stride] + \
     filter[3] * src[x + 2 * stride])

static void FUNC(put_hevc_epel_h)(int16_t *dst,
                                  uint8_t *_src, ptrdiff_t _srcstride,
                                  int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[mx - 1];
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = EPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        dst += MAX_PB_SIZE;
    }
}

static void FUNC(put_hevc_epel_v)(int16_t *dst,
                                  uint8_t *_src, ptrdiff_t _srcstride,
                                  int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[my - 1];

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = EPEL_FILTER(src, srcstride) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        dst += MAX_PB_SIZE;
    }
}

static void FUNC(put_hevc_epel_hv)(int16_t *dst,
                                   uint8_t *_src, ptrdiff_t _srcstride,
                                   int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[mx - 1];
    int16_t tmp_array[(MAX_PB_SIZE + EPEL_EXTRA) * MAX_PB_SIZE];
    int16_t *tmp = tmp_array;

    src -= EPEL_EXTRA_BEFORE * srcstride;

    for (y = 0; y < height + EPEL_EXTRA; y++)
    {
        for (x = 0; x < width; x++)
        {
            tmp[x] = EPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }

    tmp = tmp_array + EPEL_EXTRA_BEFORE * MAX_PB_SIZE;
    filter = uhd_hevc_epel_filters[my - 1];

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = EPEL_FILTER(tmp, MAX_PB_SIZE) >> 6;
        }
        tmp += MAX_PB_SIZE;
        dst += MAX_PB_SIZE;
    }
}

static void FUNC(put_hevc_epel_uni_h)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                      int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[mx - 1];
    int shift = 14 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((EPEL_FILTER(src, 1) >> (BIT_DEPTH - 8)) + offset) >> shift);
        }
        src += srcstride;
        dst += dststride;
    }
}

static void FUNC(put_hevc_epel_bi_h)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                     int16_t *src2,
                                     int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[mx - 1];
    int shift = 14 + 1 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((EPEL_FILTER(src, 1) >> (BIT_DEPTH - 8)) + src2[x] + offset) >> shift);
        }
        dst += dststride;
        src += srcstride;
        src2 += MAX_PB_SIZE;
    }
}

static void FUNC(put_hevc_epel_uni_v)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                      int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[my - 1];
    int shift = 14 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((EPEL_FILTER(src, srcstride) >> (BIT_DEPTH - 8)) + offset) >> shift);
        }
        src += srcstride;
        dst += dststride;
    }
}

static void FUNC(put_hevc_epel_bi_v)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                     int16_t *src2,
                                     int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[my - 1];
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    int shift = 14 + 1 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((EPEL_FILTER(src, srcstride) >> (BIT_DEPTH - 8)) + src2[x] + offset) >> shift);
        }
        dst += dststride;
        src += srcstride;
        src2 += MAX_PB_SIZE;
    }
}

static void FUNC(put_hevc_epel_uni_hv)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                       int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[mx - 1];
    int16_t tmp_array[(MAX_PB_SIZE + EPEL_EXTRA) * MAX_PB_SIZE];
    int16_t *tmp = tmp_array;
    int shift = 14 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    src -= EPEL_EXTRA_BEFORE * srcstride;

    for (y = 0; y < height + EPEL_EXTRA; y++)
    {
        for (x = 0; x < width; x++)
        {
            tmp[x] = EPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }

    tmp = tmp_array + EPEL_EXTRA_BEFORE * MAX_PB_SIZE;
    filter = uhd_hevc_epel_filters[my - 1];

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((EPEL_FILTER(tmp, MAX_PB_SIZE) >> 6) + offset) >> shift);
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

static void FUNC(put_hevc_epel_bi_hv)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                      int16_t *src2,
                                      int height, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[mx - 1];
    int16_t tmp_array[(MAX_PB_SIZE + EPEL_EXTRA) * MAX_PB_SIZE];
    int16_t *tmp = tmp_array;
    int shift = 14 + 1 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    src -= EPEL_EXTRA_BEFORE * srcstride;

    for (y = 0; y < height + EPEL_EXTRA; y++)
    {
        for (x = 0; x < width; x++)
        {
            tmp[x] = EPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }

    tmp = tmp_array + EPEL_EXTRA_BEFORE * MAX_PB_SIZE;
    filter = uhd_hevc_epel_filters[my - 1];

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel(((EPEL_FILTER(tmp, MAX_PB_SIZE) >> 6) + src2[x] + offset) >> shift);
            // printf("\n%d",dst[x]);
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
        src2 += MAX_PB_SIZE;
    }
    // exit(0);
}

static void FUNC(put_hevc_epel_uni_w_h)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                        int height, int denom, int wx, int ox, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[mx - 1];
    int shift = denom + 14 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    ox = ox * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel((((EPEL_FILTER(src, 1) >> (BIT_DEPTH - 8)) * wx + offset) >> shift) + ox);
        }
        dst += dststride;
        src += srcstride;
    }
}

static void FUNC(put_hevc_epel_bi_w_h)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                       int16_t *src2,
                                       int height, int denom, int wx0, int wx1,
                                       int ox0, int ox1, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[mx - 1];
    int shift = 14 + 1 - BIT_DEPTH;
    int log2Wd = denom + shift - 1;

    ox0 = ox0 * (1 << (BIT_DEPTH - 8));
    ox1 = ox1 * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
            dst[x] = uhd_clip_pixel(((EPEL_FILTER(src, 1) >> (BIT_DEPTH - 8)) * wx1 + src2[x] * wx0 +
                                     ((ox0 + ox1 + 1) << log2Wd)) >>
                                    (log2Wd + 1));
        src += srcstride;
        dst += dststride;
        src2 += MAX_PB_SIZE;
    }
}

static void FUNC(put_hevc_epel_uni_w_v)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                        int height, int denom, int wx, int ox, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[my - 1];
    int shift = denom + 14 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    ox = ox * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel((((EPEL_FILTER(src, srcstride) >> (BIT_DEPTH - 8)) * wx + offset) >> shift) + ox);
        }
        dst += dststride;
        src += srcstride;
    }
}

static void FUNC(put_hevc_epel_bi_w_v)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                       int16_t *src2,
                                       int height, int denom, int wx0, int wx1,
                                       int ox0, int ox1, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[my - 1];
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    int shift = 14 + 1 - BIT_DEPTH;
    int log2Wd = denom + shift - 1;

    ox0 = ox0 * (1 << (BIT_DEPTH - 8));
    ox1 = ox1 * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
            dst[x] = uhd_clip_pixel(((EPEL_FILTER(src, srcstride) >> (BIT_DEPTH - 8)) * wx1 + src2[x] * wx0 +
                                     ((ox0 + ox1 + 1) << log2Wd)) >>
                                    (log2Wd + 1));
        src += srcstride;
        dst += dststride;
        src2 += MAX_PB_SIZE;
    }
}

static void FUNC(put_hevc_epel_uni_w_hv)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                         int height, int denom, int wx, int ox, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[mx - 1];
    int16_t tmp_array[(MAX_PB_SIZE + EPEL_EXTRA) * MAX_PB_SIZE];
    int16_t *tmp = tmp_array;
    int shift = denom + 14 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif

    src -= EPEL_EXTRA_BEFORE * srcstride;

    for (y = 0; y < height + EPEL_EXTRA; y++)
    {
        for (x = 0; x < width; x++)
        {
            tmp[x] = EPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }

    tmp = tmp_array + EPEL_EXTRA_BEFORE * MAX_PB_SIZE;
    filter = uhd_hevc_epel_filters[my - 1];

    ox = ox * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            dst[x] = uhd_clip_pixel((((EPEL_FILTER(tmp, MAX_PB_SIZE) >> 6) * wx + offset) >> shift) + ox);
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

static void FUNC(put_hevc_epel_bi_w_hv)(uint8_t *_dst, ptrdiff_t _dststride, uint8_t *_src, ptrdiff_t _srcstride,
                                        int16_t *src2,
                                        int height, int denom, int wx0, int wx1,
                                        int ox0, int ox1, intptr_t mx, intptr_t my, int width)
{
    int x, y;
    pixel *src = (pixel *)_src;
    ptrdiff_t srcstride = _srcstride / sizeof(pixel);
    pixel *dst = (pixel *)_dst;
    ptrdiff_t dststride = _dststride / sizeof(pixel);
    const int8_t *filter = uhd_hevc_epel_filters[mx - 1];
    int16_t tmp_array[(MAX_PB_SIZE + EPEL_EXTRA) * MAX_PB_SIZE];
    int16_t *tmp = tmp_array;
    int shift = 14 + 1 - BIT_DEPTH;
    int log2Wd = denom + shift - 1;

    src -= EPEL_EXTRA_BEFORE * srcstride;

    for (y = 0; y < height + EPEL_EXTRA; y++)
    {
        for (x = 0; x < width; x++)
        {
            tmp[x] = EPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }

    tmp = tmp_array + EPEL_EXTRA_BEFORE * MAX_PB_SIZE;
    filter = uhd_hevc_epel_filters[my - 1];

    ox0 = ox0 * (1 << (BIT_DEPTH - 8));
    ox1 = ox1 * (1 << (BIT_DEPTH - 8));
    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
            dst[x] = uhd_clip_pixel(((EPEL_FILTER(tmp, MAX_PB_SIZE) >> 6) * wx1 + src2[x] * wx0 +
                                     ((ox0 + ox1 + 1) << log2Wd)) >>
                                    (log2Wd + 1));
        tmp += MAX_PB_SIZE;
        dst += dststride;
        src2 += MAX_PB_SIZE;
    }
} // line zero
#define P3 pix[-4 * xstride]
#define P2 pix[-3 * xstride]
#define P1 pix[-2 * xstride]
#define P0 pix[-1 * xstride]
#define Q0 pix[0 * xstride]
#define Q1 pix[1 * xstride]
#define Q2 pix[2 * xstride]
#define Q3 pix[3 * xstride]

// line three. used only for deblocking decision
#define TP3 pix[-4 * xstride + 3 * ystride]
#define TP2 pix[-3 * xstride + 3 * ystride]
#define TP1 pix[-2 * xstride + 3 * ystride]
#define TP0 pix[-1 * xstride + 3 * ystride]
#define TQ0 pix[0 * xstride + 3 * ystride]
#define TQ1 pix[1 * xstride + 3 * ystride]
#define TQ2 pix[2 * xstride + 3 * ystride]
#define TQ3 pix[3 * xstride + 3 * ystride]

static void FUNC(hevc_loop_filter_luma)(uint8_t *_pix,
                                        ptrdiff_t _xstride, ptrdiff_t _ystride,
                                        int beta, int *_tc,
                                        uint8_t *_no_p, uint8_t *_no_q)
{
    int d, j;
    pixel *pix = (pixel *)_pix;
    ptrdiff_t xstride = _xstride / sizeof(pixel);
    ptrdiff_t ystride = _ystride / sizeof(pixel);

    beta <<= BIT_DEPTH - 8;

    for (j = 0; j < 2; j++)
    {
        const int dp0 = abs(P2 - 2 * P1 + P0);
        const int dq0 = abs(Q2 - 2 * Q1 + Q0);
        const int dp3 = abs(TP2 - 2 * TP1 + TP0);
        const int dq3 = abs(TQ2 - 2 * TQ1 + TQ0);
        const int d0 = dp0 + dq0;
        const int d3 = dp3 + dq3;
        const int tc = _tc[j] << (BIT_DEPTH - 8);
        const int no_p = _no_p[j];
        const int no_q = _no_q[j];

        if (d0 + d3 >= beta)
        {
            pix += 4 * ystride;
            continue;
        }
        else
        {
            const int beta_3 = beta >> 3;
            const int beta_2 = beta >> 2;
            const int tc25 = ((tc * 5 + 1) >> 1);

            if (abs(P3 - P0) + abs(Q3 - Q0) < beta_3 && abs(P0 - Q0) < tc25 &&
                abs(TP3 - TP0) + abs(TQ3 - TQ0) < beta_3 && abs(TP0 - TQ0) < tc25 &&
                (d0 << 1) < beta_2 && (d3 << 1) < beta_2)
            {
                // strong filtering
                const int tc2 = tc << 1;
                for (d = 0; d < 4; d++)
                {
                    const int p3 = P3;
                    const int p2 = P2;
                    const int p1 = P1;
                    const int p0 = P0;
                    const int q0 = Q0;
                    const int q1 = Q1;
                    const int q2 = Q2;
                    const int q3 = Q3;
                    if (!no_p)
                    {
                        P0 = p0 + uhd_clip(((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) - p0, -tc2, tc2);
                        P1 = p1 + uhd_clip(((p2 + p1 + p0 + q0 + 2) >> 2) - p1, -tc2, tc2);
                        P2 = p2 + uhd_clip(((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) - p2, -tc2, tc2);
                    }
                    if (!no_q)
                    {
                        Q0 = q0 + uhd_clip(((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3) - q0, -tc2, tc2);
                        Q1 = q1 + uhd_clip(((p0 + q0 + q1 + q2 + 2) >> 2) - q1, -tc2, tc2);
                        Q2 = q2 + uhd_clip(((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) - q2, -tc2, tc2);
                    }
                    pix += ystride;
                }
            }
            else // normal filtering
            {
                int nd_p = 1;
                int nd_q = 1;
                const int tc_2 = tc >> 1;
                if (dp0 + dp3 < ((beta + (beta >> 1)) >> 3))
                {
                    nd_p = 2;
                }
                if (dq0 + dq3 < ((beta + (beta >> 1)) >> 3))
                {
                    nd_q = 2;
                }

                for (d = 0; d < 4; d++)
                {
                    const int p2 = P2;
                    const int p1 = P1;
                    const int p0 = P0;
                    const int q0 = Q0;
                    const int q1 = Q1;
                    const int q2 = Q2;
                    int delta0 = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
                    if (abs(delta0) < 10 * tc)
                    {
                        delta0 = uhd_clip(delta0, -tc, tc);
                        if (!no_p)
                        {
                            P0 = uhd_clip_pixel(p0 + delta0);
                        }
                        if (!no_q)
                        {
                            Q0 = uhd_clip_pixel(q0 - delta0);
                        }
                        if (!no_p && nd_p > 1)
                        {
                            const int deltap1 = uhd_clip((((p2 + p0 + 1) >> 1) - p1 + delta0) >> 1, -tc_2, tc_2);
                            P1 = uhd_clip_pixel(p1 + deltap1);
                        }
                        if (!no_q && nd_q > 1)
                        {
                            const int deltaq1 = uhd_clip((((q2 + q0 + 1) >> 1) - q1 - delta0) >> 1, -tc_2, tc_2);
                            Q1 = uhd_clip_pixel(q1 + deltaq1);
                        }
                    }
                    pix += ystride;
                }
            }
        }
    }
}

static void FUNC(hevc_loop_filter_chroma)(uint8_t *_pix, ptrdiff_t _xstride,
                                          ptrdiff_t _ystride, int *_tc,
                                          uint8_t *_no_p, uint8_t *_no_q)
{
    int d, j, no_p, no_q;
    pixel *pix = (pixel *)_pix;
    ptrdiff_t xstride = _xstride / sizeof(pixel);
    ptrdiff_t ystride = _ystride / sizeof(pixel);

    for (j = 0; j < 2; j++)
    {
        const int tc = _tc[j] << (BIT_DEPTH - 8);
        if (tc <= 0)
        {
            pix += 4 * ystride;
            continue;
        }
        no_p = _no_p[j];
        no_q = _no_q[j];

        for (d = 0; d < 4; d++)
        {
            int delta0;
            const int p1 = P1;
            const int p0 = P0;
            const int q0 = Q0;
            const int q1 = Q1;
            delta0 = uhd_clip((((q0 - p0) * 4) + p1 - q1 + 4) >> 3, -tc, tc);
            if (!no_p)
            {
                P0 = uhd_clip_pixel(p0 + delta0);
            }
            if (!no_q)
            {
                Q0 = uhd_clip_pixel(q0 - delta0);
            }
            pix += ystride;
        }
    }
}

static void FUNC(hevc_h_loop_filter_chroma)(uint8_t *pix, ptrdiff_t stride,
                                            int32_t *tc, uint8_t *no_p,
                                            uint8_t *no_q)
{
    FUNC(hevc_loop_filter_chroma)
    (pix, stride, sizeof(pixel), tc, no_p, no_q);
}

static void FUNC(hevc_v_loop_filter_chroma)(uint8_t *pix, ptrdiff_t stride,
                                            int32_t *tc, uint8_t *no_p,
                                            uint8_t *no_q)
{
    FUNC(hevc_loop_filter_chroma)
    (pix, sizeof(pixel), stride, tc, no_p, no_q);
}

static void FUNC(hevc_h_loop_filter_luma)(uint8_t *pix, ptrdiff_t stride,
                                          int beta, int32_t *tc, uint8_t *no_p,
                                          uint8_t *no_q)
{
    FUNC(hevc_loop_filter_luma)
    (pix, stride, sizeof(pixel),
     beta, tc, no_p, no_q);
}

static void FUNC(hevc_v_loop_filter_luma)(uint8_t *pix, ptrdiff_t stride,
                                          int beta, int32_t *tc, uint8_t *no_p,
                                          uint8_t *no_q)
{
    FUNC(hevc_loop_filter_luma)
    (pix, sizeof(pixel), stride,
     beta, tc, no_p, no_q);
}

#undef P3
#undef P2
#undef P1
#undef P0
#undef Q0
#undef Q1
#undef Q2
#undef Q3

#undef TP3
#undef TP2
#undef TP1
#undef TP0
#undef TQ0
#undef TQ1
#undef TQ2
#undef TQ3

#endif // BIT_DEPTH < 16
