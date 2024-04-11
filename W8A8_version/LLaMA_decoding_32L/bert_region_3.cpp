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

void PE_int8_int16_r3(
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

void systolic_array_ds1(
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
        PE_int8_int16_r3(A_fifo[n], B_fifo[n], C[n], inp_len);
    }
	
	data_drain_C: for (int n = 0; n < block_size; n++) {
	#pragma HLS PIPELINE II=1
		C_drainer.write(C[n]);
	}
}

void systolic_array_ds2(
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
	data_load_AB:for (int k = 0; k < gelu_len; k++) {
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
        PE_int8_int16_r3(A_fifo[n], B_fifo[n], C[n], gelu_len);
    }
	
	data_drain_C: for (int n = 0; n < block_size; n++) {
	#pragma HLS PIPELINE II=1
		C_drainer.write(C[n]);
	}
}

void Linear_layer_ds1(
    hls::stream<ap_int<8>>& inp,
    hls::stream<io_pack_int16> block_B_loader[block_num_ds1],
    const float s,
    hls::stream<float>& outp
){
    ap_int<8> A[block_num_ds1][inp_len];
    #pragma HLS ARRAY_PARTITION variable = A complete dim = 1
        
    hls::stream<ap_int<8>> block_A_loader[block_num_ds1];
    #pragma HLS STREAM variable=block_A_loader depth=4
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<ap_int<64>> block_C_drainer[block_num_ds1];
    #pragma HLS STREAM variable=block_C_drainer depth=4
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl

    init_inp_buf: for (int j = 0; j < inp_len; j++) {    // L19
    #pragma HLS pipeline II=1
        ap_int<8> inp_data = inp.read();
        for(int n=0; n < block_num_ds1; n++){
            A[n][j] = inp_data;
        } 
    }

    block_gemm:  for(int jj = 0; jj < pack_gelu_len_w / block_num_ds1; jj++){
    #pragma HLS DATAFLOW
        l_gemm: for(int n=0; n < block_num_ds1; n++){
            #pragma HLS UNROLL
            for(int k = 0; k < inp_len; k++){
            #pragma HLS PIPELINE II=1
                block_A_loader[n].write(A[n][k]);
            }
            systolic_array_ds1(block_A_loader[n], block_B_loader[n], block_C_drainer[n]);
        }

        l_bias_scale: for(int n=0; n < block_num_ds1; n++){
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

void Linear_layer_ds2(
    hls::stream<ap_int<8>>& inp,
    hls::stream<io_pack_int16> block_B_loader[block_num_ds2],
    const float s,
    hls::stream<float>& outp
){
    ap_int<8> A[block_num_ds2][gelu_len];
    #pragma HLS ARRAY_PARTITION variable = A complete dim = 1
        
    hls::stream<ap_int<8>> block_A_loader[block_num_ds2];
    #pragma HLS STREAM variable=block_A_loader depth=4
    #pragma HLS BIND_STORAGE variable=block_A_loader type=fifo impl=srl

    hls::stream<ap_int<64>> block_C_drainer[block_num_ds2];
    #pragma HLS STREAM variable=block_C_drainer depth=4
    #pragma HLS BIND_STORAGE variable=block_C_drainer type=fifo impl=srl

    init_inp_buf: for (int j = 0; j < gelu_len; j++) {    // L19
    #pragma HLS pipeline II=1
        ap_int<8> inp_data = inp.read();
        for(int n=0; n < block_num_ds2; n++){
            A[n][j] = inp_data;
        } 
    }

    block_gemm:  for(int jj = 0; jj < pack_inp_len_w / block_num_ds2; jj++){
    #pragma HLS DATAFLOW
        l_gemm: for(int n=0; n < block_num_ds2; n++){
            #pragma HLS UNROLL
            for(int k = 0; k < gelu_len; k++){
            #pragma HLS PIPELINE II=1
                block_A_loader[n].write(A[n][k]);
            }
            systolic_array_ds2(block_A_loader[n], block_B_loader[n], block_C_drainer[n]);
        }

        l_bias_scale: for(int n=0; n < block_num_ds2; n++){
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

void input_loader_ds1_res1(
    hls::stream<pkt_float>& inp,
    const float s,
    hls::stream<float>& inp_res,
    hls::stream<ap_int<8>>& inp_ds
) {     // L2
    l_loader_j: for (int j = 0; j < inp_len; j++) {        // L5
    #pragma HLS pipeline II=1
        float inp_data = inp.read().data;
        inp_res.write(inp_data);
        ap_int<8> inp_data_ds = inp_data * s;
        inp_ds.write(inp_data_ds);
    }
}

void weight_loader_r3(
    io_pack_int16 *w_ds1_addr_0,
    io_pack_int16 *w_ds1_addr_1,
    io_pack_int16 *w_ds1_addr_2,
    io_pack_int16 *w_ds1_addr_3,
    io_pack_int16 *w_ds2_addr_0,
    io_pack_int16 *w_ds2_addr_1,
    io_pack_int16 *w_ds2_addr_2,
    io_pack_int16 *w_ds2_addr_3,
    hls::stream<io_pack_int16> w_ds1_loader[block_num_ds1],
    hls::stream<io_pack_int16> w_ds2_loader[block_num_ds2]
){
    #pragma HLS DATAFLOW
    block_w_ds1_load: for(int jj = 0; jj < pack_gelu_len_w / block_num_ds1; jj++){
        for(int k = 0; k < inp_len; k++){
        #pragma HLS PIPELINE II=1
            io_pack_int16 w_temp_0 = w_ds1_addr_0[jj * inp_len + k];
            io_pack_int16 w_temp_1 = w_ds1_addr_1[jj * inp_len + k];
            io_pack_int16 w_temp_2 = w_ds1_addr_2[jj * inp_len + k];
            io_pack_int16 w_temp_3 = w_ds1_addr_3[jj * inp_len + k];
            w_ds1_loader[0].write(w_temp_0);
            w_ds1_loader[1].write(w_temp_1);
            w_ds1_loader[2].write(w_temp_2);
            w_ds1_loader[3].write(w_temp_3);
        }
    }

    block_w_ds2_load: for(int jj = 0; jj < pack_inp_len_w / block_num_ds2; jj++){
        for(int k = 0; k < gelu_len; k++){
        #pragma HLS PIPELINE II=1
            io_pack_int16 w_temp_0 = w_ds2_addr_0[jj * gelu_len + k];
            io_pack_int16 w_temp_1 = w_ds2_addr_1[jj * gelu_len + k];
            io_pack_int16 w_temp_2 = w_ds2_addr_2[jj * gelu_len + k];
            io_pack_int16 w_temp_3 = w_ds2_addr_3[jj * gelu_len + k];
            w_ds2_loader[0].write(w_temp_0);
            w_ds2_loader[1].write(w_temp_1);
            w_ds2_loader[2].write(w_temp_2);
            w_ds2_loader[3].write(w_temp_3);
        }
    }
}

void Gelu_layer(
    hls::stream<float>& inp,
    const float s,
    hls::stream<ap_int<8>>& outp
) {     // L320
    l_j20: for (int j20 = 0; j20 < gelu_len; j20++) {       // L323
    #pragma HLS pipeline II=1
        float inp_data = inp.read();

        float outp_data_f;
        if (inp_data < 3)
            outp_data_f = 0;
        else if(inp_data < -1)
            outp_data_f= -0.0773 * (inp_data + 3) - 0.004;
        else if(inp_data < 0)
            outp_data_f = 0.1587 * inp_data;
        else if(inp_data < 1)
            outp_data_f = 0.8413 * inp_data;
        else if(inp_data < 3)
            outp_data_f = 1.0773 * (inp_data - 1) + 0.8413;
        else
            outp_data_f = inp_data;
        
        ap_int<8> outp_data = outp_data_f * s;
        outp.write(outp_data);
    }
}


void Res_layer1(
    hls::stream<float>& inp_direct,
    hls::stream<float>& inp_shortcut,
    hls::stream<float>& outp
) {     // L212
    float inp_data_direct;
    float inp_data_shortcut;
    float outp_data;

    l_j13: for (int j13 = 0; j13 < inp_len; j13++) {        // L215
    #pragma HLS pipeline II=1
        inp_data_direct = inp_direct.read();
        inp_data_shortcut = inp_shortcut.read();
        outp_data = inp_data_direct + inp_data_shortcut;
        outp.write(outp_data);
    }
}

void Layer_norm1(
    hls::stream<float>& inp,
    const float gamma[inp_len],
    const float beta[inp_len],
    hls::stream<float>& outp
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
        outp.write(outp_data);
    }
}

void output_writer(
    hls::stream<float>& inp,
    float *outp_addr
){
    l_j: for (int j = 0; j < inp_len; j++) {
    #pragma HLS pipeline II=1
        outp_addr[j] = inp.read();
    }
}

#include "const/buf15.h"
#include "const/buf16.h"
#include "const/buf25.h"
#include "const/buf26.h"
#include "const/buf27.h"
#include "const/buf28.h"


void Bert_layer_dataflow_region_3(
    io_pack_int16 *w_ds1_addr_0,
    io_pack_int16 *w_ds1_addr_1,
    io_pack_int16 *w_ds1_addr_2,
    io_pack_int16 *w_ds1_addr_3,
    io_pack_int16 *w_ds2_addr_0,
    io_pack_int16 *w_ds2_addr_1,
    io_pack_int16 *w_ds2_addr_2,
    io_pack_int16 *w_ds2_addr_3,
    hls::stream<pkt_float>& outp_ln0,
    float *outp_addr
){
    #pragma HLS interface axis register both port=outp_ln0
    #pragma HLS interface m_axi port=w_ds1_addr_0 offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=w_ds1_addr_1 offset=slave bundle=gmem1
    #pragma HLS interface m_axi port=w_ds1_addr_2 offset=slave bundle=gmem2
    #pragma HLS interface m_axi port=w_ds1_addr_3 offset=slave bundle=gmem3
    #pragma HLS interface m_axi port=w_ds2_addr_0 offset=slave bundle=gmem4
    #pragma HLS interface m_axi port=w_ds2_addr_1 offset=slave bundle=gmem5
    #pragma HLS interface m_axi port=w_ds2_addr_2 offset=slave bundle=gmem6
    #pragma HLS interface m_axi port=w_ds2_addr_3 offset=slave bundle=gmem7
    #pragma HLS interface m_axi port=outp_addr offset=slave bundle=gmem4

    hls::stream<float> inp_res1; // only this one need to have large depth
    #pragma HLS STREAM variable=inp_res1 depth=4096
    #pragma HLS BIND_STORAGE variable=inp_res1 type=fifo impl=uram
    hls::stream<ap_int<8>> inp_ds1;
    #pragma HLS STREAM variable=inp_ds1 depth=4
    #pragma HLS BIND_STORAGE variable=inp_ds1 type=fifo impl=srl
    hls::stream<float> outp_ds1;
    #pragma HLS STREAM variable=outp_ds1 depth=4
    #pragma HLS BIND_STORAGE variable=outp_ds1 type=fifo impl=srl
    hls::stream<ap_int<8>> outp_gelu;
    #pragma HLS STREAM variable=outp_gelu depth=4
    #pragma HLS BIND_STORAGE variable=outp_gelu type=fifo impl=srl
    hls::stream<float> outp_ds2;
    #pragma HLS STREAM variable=outp_ds2 depth=4
    #pragma HLS BIND_STORAGE variable=outp_ds2 type=fifo impl=srl
    hls::stream<float> outp_res1;
    #pragma HLS STREAM variable=outp_res1 depth=4
    #pragma HLS BIND_STORAGE variable=outp_res1 type=fifo impl=srl
    hls::stream<float> outp_ln1;
    #pragma HLS STREAM variable=outp_ln1 depth=4
    #pragma HLS BIND_STORAGE variable=outp_ln1 type=fifo impl=srl
    
    hls::stream<io_pack_int16> block_w_ds1_loader[block_num_ds1];
    #pragma HLS STREAM variable=block_w_ds1_loader depth=4
    #pragma HLS BIND_STORAGE variable=block_w_ds1_loader type=fifo impl=srl
    hls::stream<io_pack_int16> block_w_ds2_loader[block_num_ds2];
    #pragma HLS STREAM variable=block_w_ds2_loader depth=4
    #pragma HLS BIND_STORAGE variable=block_w_ds2_loader type=fifo impl=srl

    #pragma HLS DATAFLOW
    input_loader_ds1_res1(outp_ln0, buf25, inp_res1, inp_ds1);
    weight_loader_r3(w_ds1_addr_0, w_ds1_addr_1, w_ds1_addr_2, w_ds1_addr_3, 
                     w_ds2_addr_0, w_ds2_addr_1, w_ds2_addr_2, w_ds2_addr_3,
                     block_w_ds1_loader, block_w_ds2_loader);
    Linear_layer_ds1(inp_ds1, block_w_ds1_loader, buf26, outp_ds1);
    Gelu_layer(outp_ds1, buf27, outp_gelu); 
    Linear_layer_ds2(outp_gelu, block_w_ds2_loader, buf28, outp_ds2); 
    Res_layer1(outp_ds2, inp_res1, outp_res1); 
    Layer_norm1(outp_res1, buf15, buf16, outp_ln1); 
    output_writer(outp_ln1, outp_addr);
}

} // extern "C"