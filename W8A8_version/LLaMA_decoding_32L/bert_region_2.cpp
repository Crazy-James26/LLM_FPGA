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

void PE_int8_int16_r2(
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

void systolic_array_attn(
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
	data_load_AB:for (int k = 0; k < head_len; k++) {
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
    PE_int8_int16_r2(A_fifo[n], B_fifo[n], C[n], head_len);
  }
	
	data_drain_C: for (int n = 0; n < block_size; n++) {
	#pragma HLS PIPELINE II=1
		C_drainer.write(C[n]);
	}
}

void systolic_array_cont(
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
	data_load_AB:for (int k = 0; k < seq_num; k++) {
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
        PE_int8_int16_r2(A_fifo[n], B_fifo[n], C[n], seq_num);
    }
        
    data_drain_C: for (int n = 0; n < block_size; n++) {
    #pragma HLS PIPELINE II=1
        C_drainer.write(C[n]);
    }
}


void systolic_array_ds0(
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
        PE_int8_int16_r2(A_fifo[n], B_fifo[n], C[n], inp_len);
    }
	
	data_drain_C: for (int n = 0; n < block_size; n++) {
	#pragma HLS PIPELINE II=1
		C_drainer.write(C[n]);
	}
}

void Attention_layer(
    hls::stream<ap_int<8>>& inp,
    hls::stream<io_pack_int16>& block_B_loader, 
    const float s,
    hls::stream<float>& outp,
    int head_id
){
    ap_int<8> A[head_len];

    hls::stream<ap_int<8>> block_A_loader;
    #pragma HLS STREAM variable=block_A_loader depth=4
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<ap_int<64>> block_C_drainer;
    #pragma HLS STREAM variable=block_C_drainer depth=4
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl
    
    init_inp_buf: for (int j = 0; j < head_len; j++) {    // L19
    #pragma HLS pipeline II=1
        A[j] = inp.read();
    }

    block_gemm:
    for(int jj = 0; jj < pack_seq_num_w; jj++){
    #pragma HLS DATAFLOW

        init_block_AB:
        for(int k = 0; k < head_len; k++){
        #pragma HLS PIPELINE II=1
            block_A_loader.write(A[k]);
        }

        systolic_array_attn(block_A_loader, block_B_loader, block_C_drainer);

        l_bias_scale_j: for (int j = 0; j < block_size; j++) {    // L41
        #pragma HLS pipeline II=2
            ap_int<64> outp_temp = block_C_drainer.read();
            ap_int<32> outp0_dp = outp_temp.range(31, 0);
            ap_int<32> outp1_dp = outp_temp.range(63, 32);
            float outp0 = outp0_dp / 11.3137 * s;
            float outp1 = outp1_dp / 11.3137 * s;
            outp.write(outp0);
            outp.write(outp1);
        }
    }
}


void Context_layer(
    hls::stream<ap_int<8>>& inp, 
    hls::stream<io_pack_int16>& block_B_loader, 
    const float s,
    hls::stream<ap_int<8>>& outp,
    int head_id
){
    ap_int<8> A[seq_num]; 

    hls::stream<ap_int<8>> block_A_loader;
    #pragma HLS STREAM variable=block_A_loader depth=4
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<ap_int<64>> block_C_drainer;
    #pragma HLS STREAM variable=block_C_drainer depth=4
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl

    init_inp_buf: for (int j = 0; j < seq_num; j++) {    // L19
    #pragma HLS pipeline II=1
        A[j] = inp.read();
    }

    block_gemm:
    for(int jj = 0; jj < pack_head_len_w; jj++){
    #pragma HLS DATAFLOW

        init_block_AB:
        for(int k = 0; k < seq_num; k++){
        #pragma HLS PIPELINE II=1
            block_A_loader.write(A[k]);
        }

        systolic_array_cont(block_A_loader, block_B_loader, block_C_drainer);

        l_bias_scale: for (int j = 0; j < block_size; j++) {    // L41
        #pragma HLS pipeline II=2
            ap_int<64> outp_temp = block_C_drainer.read();
            ap_int<32> outp0_dp = outp_temp.range(31, 0);
            ap_int<32> outp1_dp = outp_temp.range(63, 32);
            ap_int<8> outp0 = outp0_dp * s;
            ap_int<8> outp1 = outp1_dp * s;
            outp.write(outp0);
            outp.write(outp1);
        }
    }
}

void Softmax_layer(
    hls::stream<float>& inp,
    const float s,
    hls::stream<ap_int<8>>& outp,
    int seq_id
) {     // L86
    float buf[seq_num];
    #pragma HLS array_partition variable=buf cyclic factor=4 dim=1

    float inp_sumRow = 0;
    float partial_sum[4] = {0};

    float inp_data;
    l_buf: for (int j = 0; j < seq_num; j++) {        // L234
    #pragma HLS pipeline II=1
        inp_data = inp.read();
        if(j > seq_id)
            buf[j] = inp_data;
        else
            buf[j] = -1e10;
    }

    l_exp_sum_j6: for (int j6 = 0; j6 < seq_num/4; j6++) {     // L92
    #pragma HLS pipeline II=1
        l_exp_sum_i6: for(int i6 = 0; i6 < 4; i6++){
            float inp_data_exp = exp(buf[j6 * 4 + i6]);     
            partial_sum[i6] += inp_data_exp; 
            buf[j6 * 4 + i6] = inp_data_exp; 
        }   
    }
    
    inp_sumRow = partial_sum[0] + partial_sum[1] + partial_sum[2] + partial_sum[3];

    ap_int<8> outp_data;
    l_j7: for (int j7 = 0; j7 < seq_num; j7++) {     // L103
        #pragma HLS pipeline II=1
        outp_data = buf[j7] / inp_sumRow * s;    
        outp.write(outp_data);
    }
}

void head_spliter(
    hls::stream<pkt_int8>& inp,
    hls::stream<ap_int<8>> outp[head_parallel]
){
    ap_int<8> buf[head_num][head_len];
    #pragma HLS array_partition variable=buf cyclic factor=4 dim=1 

    l_write: for (int i = 0; i < head_num; i++) {
        for (int j = 0; j < head_len; j++) {
        #pragma HLS pipeline II=1
            buf[i][j]=inp.read().data;
        }
    }

    l_split: for(int i = 0; i < head_num / head_parallel; i++){
        for (int j = 0; j < head_len; j++) {
        #pragma HLS pipeline II=1
            for(int k = 0; k < head_parallel; k++){
                outp[k].write(buf[i * head_parallel + k][j]);
            }
        }
    }
}

void head_merger(
    hls::stream<ap_int<8>> inp[head_parallel],
    hls::stream<ap_int<8>>& outp
){
    ap_int<8> buf[head_num][head_len];
    #pragma HLS array_partition variable=buf cyclic factor=4 dim=1 

    l_buf: for(int i = 0; i < head_num / head_parallel; i++){
        for (int j = 0; j < head_len; j++) {
        #pragma HLS pipeline II=1
            for(int k = 0; k < head_parallel; k++){
                buf[i * head_parallel + k][j] = inp[k].read();
            }
        }
    }

    l_merge: for (int i = 0; i < head_num; i++) {
        for (int j = 0; j < head_len; j++) {
        #pragma HLS pipeline II=1
            outp.write(buf[i][j]);
        }
    }
}

void weight_sfa_loader(
    io_pack_int16 K[pack_seq_num_w][head_num][head_len],
    io_pack_int16 V[seq_num][head_num][pack_head_len_w],
    hls::stream<io_pack_int16> w_attn_loader[head_parallel],
    hls::stream<io_pack_int16> w_cont_loader[head_parallel]
){
    #pragma HLS DATAFLOW
    block_w_attn_load: for (int hh = 0; hh < head_num / head_parallel; hh++){
        for(int jj = 0; jj < pack_seq_num_w; jj++){
            for(int k = 0; k < head_len; k++){
                #pragma HLS PIPELINE II=1
                for (int h = 0; h < head_parallel; h++){
                    w_attn_loader[h].write(K[jj][hh * head_parallel + h][k]);
                }
            }
        }
    }

    block_w_cont_load: for (int hh = 0; hh < head_num / head_parallel; hh++){
        for(int jj = 0; jj < pack_head_len_w; jj++){
            for(int k = 0; k < seq_num; k++){
                #pragma HLS PIPELINE II=1
                for (int h = 0; h < head_parallel; h++){
                    w_cont_loader[h].write(V[k][hh * head_parallel + h][jj]);
                }
            }
        }
    }
}

void Self_attention(
    hls::stream<ap_int<8>> head_inp[head_parallel],
    hls::stream<io_pack_int16> w_attn_loader[head_parallel],
    hls::stream<io_pack_int16> w_cont_loader[head_parallel],
    const float s_attn,
    const float s_sfm,
    const float s_cont,
    hls::stream<ap_int<8>> head_outp[head_parallel],
    int seq_id
) {     // L148
    hls::stream<float> attn_outp[head_parallel];
    #pragma HLS STREAM variable=attn_outp depth=4
    #pragma HLS BIND_STORAGE variable=attn_outp type=fifo impl=srl
    hls::stream<ap_int<8>> sfm_outp[head_parallel];
    #pragma HLS STREAM variable=sfm_outp depth=4
    #pragma HLS BIND_STORAGE variable=sfm_outp type=fifo impl=srl

    l_multi_head: for (int hh = 0; hh < head_num / head_parallel; hh++){
    #pragma HLS DATAFLOW
        for (int h = 0; h < head_parallel; h++){
        #pragma HLS UNROLL
            Attention_layer(head_inp[h], w_attn_loader[h], s_attn, attn_outp[h], hh * head_parallel + h);
            Softmax_layer(attn_outp[h], s_sfm, sfm_outp[h], seq_id);
            Context_layer(sfm_outp[h], w_cont_loader[h], s_cont, head_outp[h], hh * head_parallel + h);
        }
    }
}

void weight_ds0_loader(
    io_pack_int16 *w_ds0_addr_0,
    io_pack_int16 *w_ds0_addr_1,
    hls::stream<io_pack_int16> w_ds0_loader[block_num_ds0]
){
    #pragma HLS DATAFLOW
    block_wk_load: for(int jj = 0; jj < pack_inp_len_w / block_num_ds0; jj++){
        for(int k = 0; k < inp_len; k++){
            #pragma HLS PIPELINE II=1
            io_pack_int16 w_temp_0 = w_ds0_addr_0[jj * inp_len + k];
            io_pack_int16 w_temp_1 = w_ds0_addr_1[jj * inp_len + k];
            w_ds0_loader[0].write(w_temp_0);
            w_ds0_loader[1].write(w_temp_1);
        }
    }
}

void Linear_layer_ds0(
    hls::stream<ap_int<8>>& inp,
    hls::stream<io_pack_int16> block_B_loader[block_num_ds0],
    const float s,
    hls::stream<float>& outp
){
    ap_int<8> A[inp_len];
        
    hls::stream<ap_int<8>> block_A_loader[block_num_ds0];
    #pragma HLS STREAM variable=block_A_loader depth=4
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<ap_int<64>> block_C_drainer[block_num_ds0];
    #pragma HLS STREAM variable=block_C_drainer depth=4
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl

    init_inp_buf: for (int j = 0; j < inp_len; j++) {    // L19
    #pragma HLS pipeline II=1
        A[j] = inp.read();
    }

    block_gemm:  for(int jj = 0; jj < pack_inp_len_w / block_num_ds0; jj++){
    #pragma HLS DATAFLOW
        l_gemm: for(int n=0; n < block_num_ds0; n++){
        #pragma HLS UNROLL
        for(int k = 0; k < inp_len; k++){
        #pragma HLS PIPELINE II=1
            block_A_loader[n].write(A[k]);
        }
        systolic_array_ds0(block_A_loader[n], block_B_loader[n], block_C_drainer[n]);
        }
    
        l_bias_scale: for(int n=0; n < block_num_ds0; n++){
            for (int j = 0; j < block_size; j++) {    // L41
            #pragma HLS pipeline II=2
                ap_int<64> outp_temp = block_C_drainer[n].read();
                ap_int<32> outp0_dp = outp_temp.range(31, 0);
                ap_int<32> outp1_dp = outp_temp.range(63, 32);
                float outp0 = outp0_dp * s;
                float outp1 = outp1_dp * s;
                outp.write(outp0);
                outp.write(outp1);
            }
        }
    }
}

void K_writer(
    hls::stream<pkt_int8>& inp,
    io_pack_int16 K[pack_seq_num_w][head_num][head_len],
    int seq_id
){  
    l_write_j: for (int i = 0; i < head_num; i++) {
        for (int j = 0; j < head_len; j++) {
        #pragma HLS pipeline II=1
            ap_int<8> inp_data = inp.read().data;
            K[seq_id / w_parallel][i][j].range((seq_id % w_parallel)*8 + 7, (seq_id % w_parallel)*8) = inp_data;
        }
    }
}

void V_writer(
    hls::stream<pkt_int8>& inp,
    io_pack_int16 V[seq_num][head_num][pack_head_len_w],
    int seq_id
){
    io_pack_int16 buf;
    l_write: for (int i = 0; i < head_num; i++) {
        for (int j = 0; j < pack_head_len_w; j++) {
            for (int k = 0; k < w_parallel; k++) {
            #pragma HLS pipeline II=1
                ap_int<8> inp_data = inp.read().data;
                buf.range(k*8 + 7, k*8) = inp_data;
            }
            V[seq_id][i][j]= buf;
        }
    }
}


void Res_layer0(
  hls::stream<float>& inp_direct,
  hls::stream<pkt_float>& inp_shortcut,
  hls::stream<float>& outp
) {     // L212
    float inp_data_direct;
    float inp_data_shortcut;
    float outp_data;

    l_j13: for (int j13 = 0; j13 < inp_len; j13++) {        // L215
    #pragma HLS pipeline II=1
        inp_data_direct = inp_direct.read();
        inp_data_shortcut = inp_shortcut.read().data;
        outp_data = inp_data_direct + inp_data_shortcut;
        outp.write(outp_data);
    }
}

void Layer_norm0(
    hls::stream<float>& inp,
    const float gamma[inp_len],
    const float beta[inp_len],
    hls::stream<pkt_float>& outp
) {     // L224
    float buf[inp_len];

    float mean = 0;       // L226
    float var = 0;        // L232

    float inp_data;
    float outp_data;

    l_mean_var_j14: for (int j14 = 0; j14 < inp_len; j14++) {        // L234
    #pragma HLS pipeline II=1
        inp_data = inp.read();
        buf[j14] = inp_data;
        mean += inp_data; // L238
        var += inp_data * inp_data;        // L244
    }

    mean /= inp_len; 
    var = var / inp_len - mean * mean; 

    l_j16: for (int j16 = 0; j16 < inp_len; j16++) {
    #pragma HLS pipeline II=1
        outp_data = (buf[j16] - mean) / sqrt(var + 0.000010);
        outp_data = outp_data * gamma[j16] + beta[j16];
        pkt_float data_temp;
        data_temp.data = outp_data;
        outp.write(data_temp);
    }
}


#include "const/buf13.h"
#include "const/buf14.h"
#include "const/buf21.h"
#include "const/buf22.h"
#include "const/buf23.h"
#include "const/buf24.h"

void Bert_layer_dataflow_region_2(
    io_pack_int16 *w_ds0_addr_0,
    io_pack_int16 *w_ds0_addr_1,
    hls::stream<pkt_int8>& outp_k,
    hls::stream<pkt_int8>& outp_v,
    hls::stream<pkt_int8>& outp_q,
    hls::stream<pkt_float>& outp_inp,
    hls::stream<pkt_float>& outp_ln0,
    int seq_id
){
    #pragma HLS interface m_axi port=w_ds0_addr_0 offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=w_ds0_addr_1 offset=slave bundle=gmem1
    #pragma HLS interface axis register both port=outp_k
    #pragma HLS interface axis register both port=outp_v
    #pragma HLS interface axis register both port=outp_q
    #pragma HLS interface axis register both port=outp_inp
    #pragma HLS interface axis register both port=outp_ln0


    io_pack_int16 K[pack_seq_num_w][head_num][head_len];
    #pragma HLS BIND_STORAGE variable=K type=ram_2p impl=uram
    #pragma HLS array_partition variable=K cyclic factor=4 dim=2
    io_pack_int16 V[seq_num][head_num][pack_head_len_w];
    #pragma HLS BIND_STORAGE variable=V type=ram_2p impl=uram
    #pragma HLS array_partition variable=V cyclic factor=4 dim=2

    hls::stream<ap_int<8>> head_inp[head_parallel];
    #pragma HLS STREAM variable=head_inp depth=4
    #pragma HLS BIND_STORAGE variable=head_inp type=fifo impl=srl
    hls::stream<ap_int<8>> head_outp[head_parallel];
    #pragma HLS STREAM variable=head_outp depth=4
    #pragma HLS BIND_STORAGE variable=head_outp type=fifo impl=srl

    hls::stream<ap_int<8>> outp_sfa;
    #pragma HLS STREAM variable=outp_sfa depth=4
    #pragma HLS BIND_STORAGE variable=outp_sfa type=fifo impl=srl
    hls::stream<float> outp_ds0;
    #pragma HLS STREAM variable=outp_ds0 depth=4
    #pragma HLS BIND_STORAGE variable=outp_ds0 type=fifo impl=srl
    hls::stream<float> outp_res0;
    #pragma HLS STREAM variable=outp_res0 depth=4
    #pragma HLS BIND_STORAGE variable=outp_res0 type=fifo impl=srl

    hls::stream<io_pack_int16> w_attn_loader[head_parallel];
    #pragma HLS STREAM variable=w_attn_loader depth=4
    #pragma HLS BIND_STORAGE variable=w_attn_loader type=fifo impl=srl
    hls::stream<io_pack_int16> w_cont_loader[head_parallel];
    #pragma HLS STREAM variable=w_cont_loader depth=4
    #pragma HLS BIND_STORAGE variable=w_cont_loader type=fifo impl=srl
    hls::stream<io_pack_int16> block_w_ds0_loader[block_num_ds0];
    #pragma HLS STREAM variable=block_w_ds0_loader depth=4
    #pragma HLS BIND_STORAGE variable=block_w_ds0_loader type=fifo impl=srl

    #pragma HLS DATAFLOW
    K_writer(outp_k, K, seq_id);
    V_writer(outp_v, V, seq_id);
    weight_sfa_loader(K, V, w_attn_loader, w_cont_loader);
    head_spliter(outp_q, head_inp);
    Self_attention(head_inp, w_attn_loader, w_cont_loader, buf21, buf22, buf23, head_outp, seq_id);
    head_merger(head_outp, outp_sfa);
    weight_ds0_loader(w_ds0_addr_0, w_ds0_addr_1, block_w_ds0_loader);
    Linear_layer_ds0(outp_sfa, block_w_ds0_loader, buf24, outp_ds0);
    Res_layer0(outp_ds0, outp_inp, outp_res0);
    Layer_norm0(outp_res0, buf13, buf14, outp_ln0);
}

} // extern "C"