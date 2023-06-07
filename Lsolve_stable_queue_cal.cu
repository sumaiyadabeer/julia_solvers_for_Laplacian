#include<stdio.h>
#include <string>
#include <stdlib.h>
#include<time.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>
#include <chrono>

#include "../inc/b_function.h"
#include "../inc/communication.h"
#include "../inc/eta_function.h"
#include "../inc/helper.h"
#include "../inc/calculate_error.h"

// #define N 16
// #define E 48
#define THREADS_PER_BLOCK 32

#include <cuda_runtime.h>
// #include "cublas_v2.h"
#include "device_launch_parameters.h"

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std::chrono;

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    printf("%s\n",cudaGetErrorString(x)); \
    system("pause"); \
    return EXIT_FAILURE;}} while(0)


__global__ void solve(int *row_ptr, float *b_sink, float *eta, float *beta, float kappa, float *x ){
	int index = threadIdx.x + blockIdx.x * blockDim.x; 
	x[index]=(eta[index]/(row_ptr[index+1]-row_ptr[index])); 
}

// 2.5.3. Thrust norms
struct square { __device__ __host__ float operator()(float xi) { return xi*xi; } };
struct absolute { __device__ __host__ float operator()(float xi) { return abs(xi); } };
struct int_absolute { __device__ __host__ float operator()(int xi) { return abs(xi); } };
float two_norm(const thrust::device_ptr<float> &x, int N){
// Compute the x^2s and their sum: N + k reads + k+1 writes (k is a small constant).
 thrust::device_ptr<float> x_end = x + (N);
 return sqrt(thrust::reduce(
 thrust::make_transform_iterator(x, square()),
 thrust::make_transform_iterator(x_end, square())));
}
float one_norm(const thrust::device_ptr<float> &x, int N){
 thrust::device_ptr<float> x_end = x + (N);
 return thrust::reduce(
 thrust::make_transform_iterator(x, absolute()),
 thrust::make_transform_iterator(x_end, absolute()));
}
int int_one_norm(const thrust::device_ptr<int> &x, int N){
 thrust::device_ptr<int> x_end = x + (N);
 return thrust::reduce(
 thrust::make_transform_iterator(x, int_absolute()),
 thrust::make_transform_iterator(x_end, int_absolute()));
}
float infinity_norm(const thrust::device_ptr<float> &x, int N){
 thrust::device_ptr<float> x_end = x + (N);
 return max( float(thrust::max_element(x, x_end)[0]),
 			 abs(float(thrust::min_element(x, x_end)[0]))
			);
}
int main(int argc, char** argv) {
	
	int devNum = 0;
    // CUDA_CALL(cudaGetDevice(&devNum));
    CUDA_CALL(cudaSetDevice(devNum));
	printf("Code is executing on device %d \n",devNum );



	// device copies 
	int *d_row_ptr, *d_col_off, *d_values, *d_b_sum, *d_L; 
	int *d_queue, *d_outbox, *d_outbox_index, *d_outbox_count, *d_cnt, *d_stable_cnt, *d_b_sink_index, *d_sum_Q ;
	float  *d_eta, *d_eta_tminusone, *d_eta_del, *d_eta_max, *d_eta_sum, *d_eta_del_norm;
	float *d_b, *d_J, *d_b_norm, *d_b_sink,  *d_x, *d_beta; // *d_Lx_b, *d_Lx_b_norm,
	curandState *d_state;



	std::string file_path= argv[1]; //"data/generated_input_16384_random.txt";
	std::string answer_file_path="./generated_answer.txt";
	int NE[2];
	printf(".");

	read_file_by_line(file_path, NE, 0, 2);
	printf(".");
	const unsigned int N = NE[0];
	const unsigned int NN = NE[0]+32;
	const unsigned int E = NE[1]; //BECAUSE IN UNDIRECTED GRAPH EVERY EDGE IS COUNTED TWICE
	

// Alloc space for device copies of graph N E and beta epsilon and kappa eta max
	const unsigned int int_size =  sizeof(int);
	const unsigned int float_size = sizeof(float);
	

	CUDA_CALL(cudaMalloc((void **)&d_row_ptr, (NN+1)*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_b, NN*float_size));
	CUDA_CALL(cudaMalloc((void **)&d_J, NN*float_size));  
	CUDA_CALL(cudaMalloc((void **)&d_b_norm, float_size));
	CUDA_CALL(cudaMalloc((void **)&d_x, NN*float_size));
	// CUDA_CALL(cudaMalloc((void **)&d_Lx_b, N*float_size));
	// CUDA_CALL(cudaMalloc((void **)&d_Lx_b_norm, float_size));

	CUDA_CALL(cudaMalloc((void **)&d_eta_sum, float_size));
	CUDA_CALL(cudaMalloc((void **)&d_eta_max, float_size));
	// CUDA_CALL(cudaMalloc((void **)&d_eta_del_norm, float_size));
	CUDA_CALL(cudaMalloc((void **)&d_eta, NN*float_size));
	// CUDA_CALL(cudaMalloc((void **)&d_eta_del, NN*float_size));
	CUDA_CALL(cudaMalloc((void **)&d_eta_tminusone, NN*float_size));
	CUDA_CALL(cudaMalloc((void **)&d_b_sink, float_size));
	CUDA_CALL(cudaMalloc((void **)&d_b_sink_index, int_size));
	CUDA_CALL(cudaMalloc((void **)&d_sum_Q, int_size));
	CUDA_CALL(cudaMalloc((void **)&d_beta, float_size));

	CUDA_CALL(cudaMalloc((void **)&d_col_off, E*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_values, E*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_b_sum, int_size));

	CUDA_CALL(cudaMalloc((void **)&d_queue, NN*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_outbox, NN*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_outbox_index, NN*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_outbox_count, NN*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_cnt, NN*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_stable_cnt, NN*int_size));
	CUDA_CALL(cudaMalloc((void **)&d_L, NN*NN*int_size));

	CUDA_CALL(cudaMalloc((void **)&d_state, NN*sizeof(curandState)));
	

	//read the graph and b from input file

	int *row_ptr = (int*)malloc((NN+1)*int_size);
	read_file_by_line(file_path, row_ptr, 1, NN+1);
	printf(".");
	int *col_off = (int*)malloc(E*int_size);
	read_file_by_line(file_path, col_off, 2, E);
	printf(".");
	int *values = (int*)malloc(E*int_size);
	read_file_by_line(file_path, values, 3, E);
	printf(".");
	float *b = (float*)malloc(NN*float_size);
	read_file_by_line(file_path, b, 4, NN);
	printf(".");


	// this loop is for printing purpose of input values
	// for (int i=0;i<N;i++){
	// 	std :: cout<<i<<"\t"<< b[i]<<std :: endl;
	// }
	// return -1;
	

// Copy graph to device
	CUDA_CALL(cudaMemcpy(d_row_ptr, row_ptr, (NN+1)*int_size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_col_off, col_off, E*int_size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_values, values, E*int_size, cudaMemcpyHostToDevice));	
	CUDA_CALL(cudaMemcpy(d_b, b, NN*float_size, cudaMemcpyHostToDevice));
	// CUDA_CALL(cudaMemcpy(d_DJ, b, N*float_size, cudaMemcpyHostToDevice));
 
	printf("input and copy to device done \n");



	thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(d_b);
	thrust::device_ptr<int> int_dev_ptr = thrust::device_pointer_cast(d_queue);
	
	thrust::device_ptr<int> int_dev_ptr2 = thrust::device_pointer_cast(d_queue);
	thrust::device_vector<int> output_keys(N);
	thrust::device_vector<int> output_freqs(N);
	thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;		
	thrust::device_vector<int> dev_ones(N);




// Host space allocation
	float *eta = (float*)malloc(N*float_size);
	int *queue = (int*)malloc(N*int_size);
	float *rhs_norm = (float*)malloc(float_size);
	// float *Lx_b_norm = (float*)malloc(N*float_size);
	float *beta = (float*)malloc(float_size);
	float *result = (float*)malloc(N*sizeof(float));
	float *eta_del_norm = (float*)malloc(float_size); 
	float *eta_sum = (float*)malloc(float_size);
	float *eta_max = (float*)malloc(float_size);
	
	
//Initial_setup
	
	const double EPS =  1.19209e-07; //1.0/(N*N*N);
	double eta_max_threshold = 0.6; //(0.75)*(1-EPS);	see the logic in paper
	float frac_of_packet_sunk_threshold = 0.70;

	int num_of_blocks =  (N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;  
	// unsigned int max_epoch	= 100000; //100000; //should depend on graph size and topology
	unsigned int epoch;
	int sink_index;
	int T;
	int Q_sink_index;
	int sum_Q;
	float frac_of_packet_sunk;
	int eta_gt_chk_more_thn_i;
	int eta_del_lt_eps_more_thn_i;
	int frac_of_packet_sunk_more_thn_i;
	float send_recv_rounds;
	bool flag_frac_of_packet = false;

	

	*beta = 1.0;
	CUDA_CALL(cudaMemcpy(d_beta, beta, float_size, cudaMemcpyHostToDevice));

	
	get_b_sink<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_b, d_b_sink, N, d_b_sink_index); // return min vale of sink (-9)
	// CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaMemcpy(&sink_index, d_b_sink_index, int_size, cudaMemcpyDeviceToHost));
	printf("sink index : %d \n", sink_index );

	

	convert_b_to_J<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_b, N, d_b_sink);
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaMemcpy(b, d_b, N*float_size, cudaMemcpyDeviceToHost));
	convert_J_to_2betaJ<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_b, N, d_beta); //bcz this will get halved after entering the loop
	CUDA_CALL(cudaDeviceSynchronize());

	
	do{
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
  		// *beta = *beta/2;
		send_recv_rounds = *beta;
		eta_gt_chk_more_thn_i = 0;
		frac_of_packet_sunk_more_thn_i = 0;
		eta_del_lt_eps_more_thn_i = 0;
		epoch = 0;

		while (send_recv_rounds < 10.0) //this loop is to make sure to generate packet in group of epoch.. waz creating problem inn visualization
			send_recv_rounds *= 10.0;
				
		CUDA_CALL(cudaMemcpy(d_beta, beta, sizeof(float), cudaMemcpyHostToDevice));
		update_b<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_b); // returns b=\beta*b
		// CUDA_CALL(cudaMemcpy(b, d_b, N*sizeof(float), cudaMemcpyDeviceToHost));
		// CUDA_CALL(cudaDeviceSynchronize());
		// for (int i=0; i<N; i++)
		// 	printf("%f\t", b[i]);
		// printf("\n");

		initialize<<<num_of_blocks,THREADS_PER_BLOCK>>>(b, d_eta, d_cnt, d_queue, d_outbox, d_L, N, E, d_state, 2*rand()); //set eta queue outbox as 0_randomstae
		// initialize<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_eta, d_stable_cnt, d_queue, d_outbox, d_L, N, E, d_state, 2*rand()); //set eta queue outbox as 0_randomstae
		
		do{
			get_b_sink<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_b, d_b_sink, N, d_b_sink_index);
			CUDA_CALL(cudaDeviceSynchronize());
			CUDA_CALL(cudaMemcpy(beta, d_b_sink, sizeof(float), cudaMemcpyDeviceToHost));

			*beta = -*beta;
			// printf("%f \n", *beta);

			// printf("In DRW compute iter: %d beta: %f eta_max %f \n", epoch, *beta, *eta_max);
			copy_eta<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_eta, d_eta_tminusone);
			CUDA_CALL(cudaDeviceSynchronize());

			for(int i=0; i<(int)send_recv_rounds; i++){
				epoch++;
				
				// printf("%d ", epoch);
				send<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_row_ptr, d_b, d_col_off, d_values, d_queue, d_outbox, d_cnt, d_state,  rand(), rand(), E);
				// CUDA_CALL(cudaDeviceSynchronize());
				recv<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_outbox, d_queue, d_b, N);
							 
				}
			
				calculate_eta<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_eta, d_cnt, float(epoch));


			/**************Termination condition prep if queues are saturated *******************/
			CUDA_CALL(cudaDeviceSynchronize());
			dev_ptr = thrust::device_pointer_cast(d_eta);
			*eta_max = infinity_norm(dev_ptr, N);
			// CUDA_CALL(cudaDeviceSynchronize());
			// CUDA_CALL(cudaMemcpy(eta_max, d_eta_max, sizeof(float), cudaMemcpyDeviceToHost));	
			
			// printf("Epoch: %d \t eta_del_norm: %f \t eta_del_norm<=EPS: %s \t eta_del_norm>0: %s \t eta_max_inner: %f\n", epoch, *eta_del_norm, (*(eta_del_norm) <= EPS)?"T":"F", (*(eta_del_norm)>0)?"T":"F", *eta_max);
			
			
			/**************Termination condition prep for Q[sink]/(1+sum(Q)) *******************/
			CUDA_CALL(cudaMemcpy(&Q_sink_index, d_queue+sink_index, int_size, cudaMemcpyDeviceToHost));
			// printf("sink index : %d \n", sink_index );
			int_dev_ptr = thrust::device_pointer_cast(d_queue);
			sum_Q = int_one_norm(int_dev_ptr, N);
			frac_of_packet_sunk = (float)Q_sink_index/(float)(1+sum_Q);
			
			// printf(" Frac of packet sunk: %f\t sink: %d\t sum: %d\n", frac_of_packet_sunk, Q_sink_index, sum_Q);
 			

			// Termination in action
			if(0){	// ((*(eta_del_norm) <= EPS) && (*(eta_del_norm)>0))
				eta_del_lt_eps_more_thn_i++;
				if (eta_del_lt_eps_more_thn_i >= 10){
					// printf("eta_del_norm is lt threshold so breaking\n");
					// break;
				}
			}else if((*eta_max >= eta_max_threshold)){
				eta_del_lt_eps_more_thn_i = 0;
				eta_gt_chk_more_thn_i++;
				if (eta_gt_chk_more_thn_i >= 3){
					printf("eta_max is gt threshold so breaking\n");
					break;
				}

			}else if(frac_of_packet_sunk > frac_of_packet_sunk_threshold){
				eta_del_lt_eps_more_thn_i = 0;
				eta_gt_chk_more_thn_i = 0;
				
				frac_of_packet_sunk_more_thn_i++;
				if (flag_frac_of_packet != true){
					if (frac_of_packet_sunk_more_thn_i >= 5){ 
						flag_frac_of_packet = true;
						
						CUDA_CALL(cudaDeviceSynchronize());
						printf("frac_of_packet_sunk is gt threshold so breaking sink: %d\t sum: %d\t frac: %f\n", Q_sink_index, sum_Q, frac_of_packet_sunk);						
						break; 
					}
				}
					

			}else{
				eta_del_lt_eps_more_thn_i = 0;
				eta_gt_chk_more_thn_i = 0;
				frac_of_packet_sunk_more_thn_i = 0;
			}

		}while(1); //epoch < max_epoch

		high_resolution_clock::time_point t2 = high_resolution_clock::now();
  		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
			

		printf("epoch: %d \t eta_max: %f \t max_allowed_eta: %f \t fraction_of_packet_sunk: %f \t beta: %f \t duartion: ", epoch, *eta_max, eta_max_threshold, frac_of_packet_sunk, *beta);
		std::cout<<time_span.count()<<" sec"<<std::endl;
					

			
	 	CUDA_CALL(cudaMemcpy(eta, d_eta, (N)*sizeof(float), cudaMemcpyDeviceToHost));

		// printf("\nprinting eta \n");
		// for (int i=0; i<N; i++)
		// 	printf("%f\n", eta[i]);
		// printf("printing eta ends \n");

		if ((*eta_max) > eta_max_threshold && (*eta_max)>0){
			printf("going for next iteraion \n");
			continue;
		}else{

			//set the cnt and T as zero
			make_cnt_0<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_cnt);
			T = 0;
			printf("Calculating the stable queue occupancy \n");

			while(frac_of_packet_sunk < 0.80){
				for(int i=0; i<(int)send_recv_rounds; i++){
				T++;
				send<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_row_ptr, d_b, d_col_off, d_values, d_queue, d_outbox, d_cnt, d_state,  rand(), rand(), E);
				recv<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_outbox, d_queue, d_b, N);
				}			 
				// frac_of_packet_sunk update
				CUDA_CALL(cudaMemcpy(&Q_sink_index, d_queue+sink_index, int_size, cudaMemcpyDeviceToHost));
				int_dev_ptr = thrust::device_pointer_cast(d_queue);
				sum_Q = int_one_norm(int_dev_ptr, N);
				frac_of_packet_sunk = (float)Q_sink_index/(float)(1+sum_Q);	
				// printf(" sink: %d\t sum: %d\t frac: %f\n", Q_sink_index, sum_Q, frac_of_packet_sunk);						
						
			}
			//run the send and rcv till the frac_of_packet_sunk is 80%

			calculate_eta<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_eta, d_cnt, float(T));

			solve<<<num_of_blocks,THREADS_PER_BLOCK>>>(d_row_ptr, d_b_sink, d_eta,  d_beta, 0.0001 , d_x );//(int *row_ptr, float *b_sink, float *eta, float *beta, float kappa, float *x )
			CUDA_CALL(cudaMemcpy(eta, d_x, (N)*sizeof(float), cudaMemcpyDeviceToHost));
			// printf("\nprinting x \n");
			// for (int i=0; i<N; i++)
			// 	printf("%f\n", eta[i]);
			// printf("printing x ends \n");
			
			
			std::ofstream myfile (argv[2]);
			if (myfile.is_open())
			{
				printf("Wrting the results in file.. Please be patient :)\n");
				myfile << time_span.count() << "\n\n" ; // writing time and laving a line for residual

				for(int count = 0; count < N; count ++){
					myfile << eta[count] << "\n" ;
				}
				myfile.close();
			}
			else std::cout << "Unable to open file";

			break; // As now nothing else is left to do
		
		}
			
	}while(1);
	

// Cleanup
	free(row_ptr);    free(b);    free(eta);    free(col_off);    free(values);    free(rhs_norm);	free(beta);    free(result); 	free(eta_del_norm); 	free(eta_sum); 	  free(eta_max);
	cudaFree(d_row_ptr); 	cudaFree(d_b); 	cudaFree(d_J); 	cudaFree(d_b_norm); 	cudaFree(d_x); 		cudaFree(d_eta_sum);	cudaFree(d_eta_max); 	
	// cudaFree(d_eta_del_norm); 	cudaFree(d_eta_del); 	
	cudaFree(d_eta);	cudaFree(d_eta_tminusone); 	cudaFree(d_b_sink); 	cudaFree(d_b_sink_index);	cudaFree(d_beta);	cudaFree(d_col_off); 	cudaFree(d_values); 	cudaFree(d_b_sum);	cudaFree(d_queue); 	cudaFree(d_outbox);	cudaFree(d_cnt);	cudaFree(d_L);	cudaFree(&d_state);
		// free(Lx_b_norm);
		// cudaFree(d_Lx_b); 	cudaFree(d_Lx_b_norm);
	return 0;
}

