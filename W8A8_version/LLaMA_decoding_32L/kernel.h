#ifndef _GEinp_numinp_num_SYSTOLIC_ARRAY_H_
#define _GEinp_numinp_num_SYSTOLIC_ARRAY_H_

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

#define seq_num 512
#define inp_len 4096
#define head_num 32
#define head_len 128
#define gelu_len 11008

#define w_parallel 64
#define block_size 32
#define block_num_qkv 2
#define head_parallel 4
#define block_num_ds0 2
#define block_num_ds1 4
#define block_num_ds2 4

#define pack_seq_num_w seq_num/w_parallel
#define pack_inp_len_w inp_len/w_parallel
#define pack_head_len_w head_len/w_parallel
#define pack_gelu_len_w gelu_len/w_parallel

typedef ap_uint<8 * w_parallel> io_pack_int16;

typedef union {
  float f;
  uint32_t i;
} converter_t;

typedef ap_axiu<8, 0, 0, 0> pkt_int8;
typedef ap_axiu<32, 0, 0, 0> pkt_float;

#endif