#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include "kernel.h"
using namespace std;

extern "C" {
void PE_int8_int16(
  hls::stream<ap_int<8>>& A_in,
  hls::stream<ap_int<16>>& B_in,
  ap_int<64>& C_out, int k_size
) {
  ap_int<32> C_out1;
  ap_int<32> C_out0;
  PE_LOOP: for (int k = 0; k < k_size; k++) {
  #pragma HLS PIPELINE II=1
    ap_int<8> a = A_in.read();
    ap_int<16> b = B_in.read();
    ap_int<24> pack_b = ap_int<24>((b(15, 8), ap_uint<16>(0))) + ap_int<24>(b(7, 0));
    ap_int<16> c1;
    ap_int<16> c0;
    (c1, c0) = a * pack_b;
    c1 = c1 + c0[15];
    C_out1 += c1;
    C_out0 += c0;
  }
  C_out = (C_out1, C_out0);
}

void systolic_array_qkv(
  hls::stream<ap_int<8>>& A_loader, 
  hls::stream<io_pack_int16>& B_loader, 
  hls::stream<ap_int<64>>& C_drainer
) {
  hls::stream<ap_int<8>> A_fifo[block_size];
  hls::stream<ap_int<16>> B_fifo[block_size];
  #pragma HLS STREAM variable=A_fifo depth=4
  #pragma HLS BIND_STORAGE variable=A_fifo type=fifo impl=srl
  #pragma HLS STREAM variable=B_fifo depth=4
  #pragma HLS BIND_STORAGE variable=B_fifo type=fifo impl=srl

  ap_int<64> C[block_size] = {0};
  #pragma HLS ARRAY_PARTITION variable = C complete dim = 1

	#pragma HLS DATAFLOW
	data_load_AB:for (int k = 0; k < inp_len; k++) {
	#pragma HLS PIPELINE II=1
		ap_int<8> A_temp = A_loader.read();
    io_pack_int16 B_temp = B_loader.read();

		for (int m = 0; m < block_size; m++) {
			A_fifo[m].write(A_temp);
		}
		
		for (int n = 0; n < block_size; n++) {
			B_fifo[n].write(B_temp.range(n*16 + 15, n*16));
		}
	}

  for (int n = 0; n < block_size; n++) {
  #pragma HLS UNROLL
    PE_int8_int16(A_fifo[n], B_fifo[n], C[n], inp_len);
  }
	
	data_drain_C: for (int n = 0; n < block_size; n++) {
	#pragma HLS PIPELINE II=1
		C_drainer.write(C[n]);
	}
}

void Linear_layer_qkv(
  hls::stream<ap_int<8>>& inp,
  hls::stream<io_pack_int16> block_B_loader[block_num_qkv],
  const float s,
  hls::stream<pkt_int8>& outp
){
  ap_int<8> A[inp_len];
    
  hls::stream<ap_int<8>> block_A_loader[block_num_qkv];
  #pragma HLS STREAM variable=block_A_loader depth=4
  #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

  hls::stream<ap_int<64>> block_C_drainer[block_num_qkv];
  #pragma HLS STREAM variable=block_C_drainer depth=4
  #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl

  init_inp_buf: for (int j = 0; j < inp_len; j++) {    // L19
  #pragma HLS pipeline II=1
      A[j] = inp.read();
  }

  
  block_gemm:  for(int jj = 0; jj < pack_inp_len_w / block_num_qkv; jj++){
  #pragma HLS DATAFLOW
    l_gemm: for(int n=0; n < block_num_qkv; n++){
    #pragma HLS UNROLL
      for(int k = 0; k < inp_len; k++){
      #pragma HLS PIPELINE II=1
        block_A_loader[n].write(A[k]);
      }
      systolic_array_qkv(block_A_loader[n], block_B_loader[n], block_C_drainer[n]);
    }

    pkt_int8 pkt_temp;
    l_bias_scale: for(int n=0; n < block_num_qkv; n++){
      for (int j = 0; j < block_size; j++) {    // L41
      #pragma HLS pipeline II=2
        ap_int<64> outp_temp = block_C_drainer[n].read();
        ap_int<32> outp0_dp = outp_temp.range(31, 0);
        ap_int<32> outp1_dp = outp_temp.range(63, 32);
        ap_int<8> outp0 = outp0_dp * s;
        ap_int<8> outp1 = outp1_dp * s;
        pkt_temp.data = outp0;
        outp.write(pkt_temp);
        pkt_temp.data = outp1;
        outp.write(pkt_temp);
      }
    }
  }
}

void input_loader(
  float *inp_addr,
  hls::stream<pkt_float>& outp_inp
){
  pkt_float pkt_temp;
  l_load_j: for (int j = 0; j < inp_len; j++) {
  #pragma HLS pipeline II=1
    float inp = inp_addr[j];
    pkt_temp.data = inp;
    outp_inp.write(pkt_temp);
  }
}

void input_loader_kv(
  float *inp_addr,
  const float s,
  hls::stream<ap_int<8>>& inp_k,
  hls::stream<ap_int<8>>& inp_v
){
  l_load_j: for (int j = 0; j < inp_len; j++) {
  #pragma HLS pipeline II=1
    ap_int<8> inp_data = inp_addr[j] * s;
    inp_k.write(inp_data);
    inp_v.write(inp_data);
  }
}

void input_loader_q(
  float *inp_addr,
  const float s,
  hls::stream<ap_int<8>>& inp_q
){
  l_load_j: for (int j = 0; j < inp_len; j++) {
  #pragma HLS pipeline II=1
    int8_t inp_data = inp_addr[j] * s;
    inp_q.write(inp_data);
  }
}

void weight_loader_r1(
  io_pack_int16 *wk_addr_0,
  io_pack_int16 *wk_addr_1,
  io_pack_int16 *wv_addr_0,
  io_pack_int16 *wv_addr_1,
  io_pack_int16 *wq_addr_0,
  io_pack_int16 *wq_addr_1,
  hls::stream<io_pack_int16> wk_loader[block_num_qkv],
  hls::stream<io_pack_int16> wv_loader[block_num_qkv],
  hls::stream<io_pack_int16> wq_loader[block_num_qkv]
){
  #pragma HLS DATAFLOW
  block_wk_load: for(int jj = 0; jj < pack_inp_len_w / block_num_qkv; jj++){
    for(int k = 0; k < inp_len; k++){
    #pragma HLS PIPELINE II=1
      io_pack_int16 w_temp_0 = wk_addr_0[jj * inp_len + k];
      io_pack_int16 w_temp_1 = wk_addr_1[jj * inp_len + k];
      wk_loader[0].write(w_temp_0);
      wk_loader[1].write(w_temp_1);
    }
  }

  block_wv_load: for(int jj = 0; jj < pack_inp_len_w / block_num_qkv; jj++){
    for(int k = 0; k < inp_len; k++){
    #pragma HLS PIPELINE II=1
      io_pack_int16 w_temp_0 = wv_addr_0[jj * inp_len + k];
      io_pack_int16 w_temp_1 = wv_addr_1[jj * inp_len + k];
      wv_loader[0].write(w_temp_0);
      wv_loader[1].write(w_temp_1);
    }
  }

  block_wq_load: for(int jj = 0; jj < pack_inp_len_w / block_num_qkv; jj++){
    for(int k = 0; k < inp_len; k++){
    #pragma HLS PIPELINE II=1
      io_pack_int16 w_temp_0 = wq_addr_0[jj * inp_len + k];
      io_pack_int16 w_temp_1 = wq_addr_1[jj * inp_len + k];
      wq_loader[0].write(w_temp_0);
      wq_loader[1].write(w_temp_1);
    }
  }
}

#include "const/buf17.h"
#include "const/buf18.h"
#include "const/buf19.h"
#include "const/buf20.h"

void Bert_layer_dataflow_region_1(
  float *inp_addr_0,
  float *inp_addr_1,
  float *inp_addr_2,
  io_pack_int16 *wk_addr_0,
  io_pack_int16 *wk_addr_1,
  io_pack_int16 *wv_addr_0,
  io_pack_int16 *wv_addr_1,
  io_pack_int16 *wq_addr_0,
  io_pack_int16 *wq_addr_1,
  hls::stream<pkt_int8>& outp_k,
  hls::stream<pkt_int8>& outp_v,
  hls::stream<pkt_int8>& outp_q,
  hls::stream<pkt_float>& outp_inp
){
  #pragma HLS interface m_axi port=inp_addr_0 offset=slave bundle=gmem0
  #pragma HLS interface m_axi port=inp_addr_1 offset=slave bundle=gmem1
  #pragma HLS interface m_axi port=inp_addr_2 offset=slave bundle=gmem2
  #pragma HLS interface m_axi port=wk_addr_0 offset=slave bundle=gmem3
  #pragma HLS interface m_axi port=wk_addr_1 offset=slave bundle=gmem4
  #pragma HLS interface m_axi port=wv_addr_0 offset=slave bundle=gmem5
  #pragma HLS interface m_axi port=wv_addr_1 offset=slave bundle=gmem6
  #pragma HLS interface m_axi port=wq_addr_0 offset=slave bundle=gmem7
  #pragma HLS interface m_axi port=wq_addr_1 offset=slave bundle=gmem8
  #pragma HLS interface axis register both port=outp_k
  #pragma HLS interface axis register both port=outp_v
  #pragma HLS interface axis register both port=outp_q
  #pragma HLS interface axis register both port=outp_inp
 

  hls::stream<ap_int<8>> inp_k;
  #pragma HLS STREAM variable=inp_k depth=4
  #pragma HLS BIND_STORAGE variable=inp_k type=fifo impl=srl
  hls::stream<ap_int<8>> inp_v;
  #pragma HLS STREAM variable=inp_v depth=4
  #pragma HLS BIND_STORAGE variable=inp_v type=fifo impl=srl
  hls::stream<ap_int<8>> inp_q;
  #pragma HLS STREAM variable=inp_q depth=4
  #pragma HLS BIND_STORAGE variable=inp_q type=fifo impl=srl

  hls::stream<io_pack_int16> block_wk_loader[block_num_qkv];
  #pragma HLS STREAM variable=block_wk_loader depth=4
  #pragma HLS BIND_STORAGE variable=block_wk_loader type=fifo impl=srl
  hls::stream<io_pack_int16> block_wv_loader[block_num_qkv];
  #pragma HLS STREAM variable=block_wv_loader depth=4
  #pragma HLS BIND_STORAGE variable=block_wv_loader type=fifo impl=srl
  hls::stream<io_pack_int16> block_wq_loader[block_num_qkv];
  #pragma HLS STREAM variable=block_wq_loader depth=4
  #pragma HLS BIND_STORAGE variable=block_wq_loader type=fifo impl=srl
  
  #pragma HLS DATAFLOW
  input_loader(inp_addr_0, outp_inp);  // L457
  input_loader_kv(inp_addr_1, buf17, inp_k, inp_v);  // L457
  input_loader_q(inp_addr_2, buf17, inp_q);  // L457
  weight_loader_r1(wk_addr_0, wk_addr_1, wv_addr_0, wv_addr_1, wq_addr_0, wq_addr_1, block_wk_loader, block_wv_loader, block_wq_loader);
  Linear_layer_qkv(inp_k, block_wk_loader, buf18, outp_k);  // L458
  Linear_layer_qkv(inp_v, block_wv_loader, buf19, outp_v);  // L459
  Linear_layer_qkv(inp_q, block_wq_loader, buf20, outp_q);  // L460
}

} // end extern C