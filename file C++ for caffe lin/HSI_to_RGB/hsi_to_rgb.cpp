#include <cmath>
#include <math.h>    
#include <vector>

#include "caffe/layers/hsi_to_rgb.hpp"


const double pi = 3.141592654;

namespace caffe {


template <typename Dtype>
void HSI2RGBLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	N = bottom[0]->shape(0);
	C = bottom[0]->shape(1);
	H = bottom[0]->shape(2);
	W = bottom[0]->shape(3);
	output_H_ = bottom[0]->shape(2);
	output_W_ = bottom[0]->shape(3);
 }
	
	
template <typename Dtype>
void HSI2RGBLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* U = bottom[0]->cpu_data();
  Dtype* V = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < N; ++i) 
  {
	  for(int a=0; a < output_W_ * output_H_; ++a)
	  {
		 Dtype H_g = U[i*output_H_ * output_W_*C + a];
		 Dtype S = U[i*output_H_ * output_W_*C + a + output_H_ * output_W_] ;
		 Dtype I_new = U[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] ;
		 	 	 

		 if ((H_g >= 0) && (H_g < (2 * pi / 3)))
			{
				V[i*output_H_ * output_W_*C + a] = I_new*(1 - S);
				V[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] = I_new*(1 + (S*cos(H_g)) / cos(pi / 3 - H_g));
				V[i*output_H_ * output_W_*C + a + output_H_ * output_W_] = 3 * I_new - (V[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] + V[i*output_H_ * output_W_*C + a]);
			}

		 else if ((H_g >= (2 * pi / 3)) && (H_g < (4 * pi / 3)))
			{
				H_g = H_g - 2 * pi / 3;

				V[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] = I_new*(1 - S);
				V[i*output_H_ * output_W_*C + a + output_H_ * output_W_] = I_new*(1 + (S*cos(H_g)) / cos(pi / 3 - H_g));
				V[i*output_H_ * output_W_*C + a] = 3 * I_new - (V[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] + V[i*output_H_ * output_W_*C + a + output_H_ * output_W_]);

			}

		 else if ((H_g >= (4 * pi / 3)) && (H_g < (2 * pi)))
			{
				H_g = H_g - 4 * pi / 3;
				
				V[i*output_H_ * output_W_*C + a + output_H_ * output_W_] = I_new*(1 - S);
				V[i*output_H_ * output_W_*C + a] = I_new*(1 + (S*cos(H_g)) / cos(pi / 3 - H_g) );
				V[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] = 3 * I_new - (V[i*output_H_ * output_W_*C + a] + V[i*output_H_ * output_W_*C + a + output_H_ * output_W_]);

			}
	 
	  }
	 	  
   
  }
}

template <typename Dtype>
void HSI2RGBLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
 // if (propagate_down[0]) {
    
	const Dtype* U = bottom[0]->cpu_data();
    const Dtype* dV = top[0]->cpu_diff();
    Dtype* dU = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
	Dtype bottom_datum;
	
    //for (int i = 0; i < count; ++i) {
    //  bottom_datum = bottom_data[i];
    //  bottom_diff[i] = top_diff[i] * cos(bottom_datum) ;
    //}
	
	
	for (int i = 0; i < N; ++i)
	{
		for (int a = 0; a < output_W_ * output_H_; ++a)
		{

			Dtype H_g = U[i*output_H_ * output_W_*C + a];
			Dtype S = U[i*output_H_ * output_W_*C + a + output_H_ * output_W_];
			Dtype I_new = U[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_];

			if ((H_g >= 0) && (H_g < (2 * pi / 3)))
			{
				//dU[i*output_H_ * output_W_*C + a] += -dV[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] * I_new*S*sin(pi / 3) / pow(cos(pi / 3 - H_g), 2);
				//dU[i*output_H_ * output_W_*C + a] += dV[i*output_H_ * output_W_*C + a + output_H_ * output_W_] * I_new*S*sin(pi / 3) / pow(cos(pi / 3 - H_g), 2);

				//dU[i*output_H_ * output_W_*C + a + output_H_ * output_W_] += -dV[i*output_H_ * output_W_*C + a] * I_new;
				//dU[i*output_H_ * output_W_*C + a + output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] * I_new*cos(H_g) / cos(pi / 3 - H_g);
				//dU[i*output_H_ * output_W_*C + a + output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a + output_H_ * output_W_] * I_new*(1 - cos(H_g) / cos(pi / 3 - H_g));

				dU[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a] * (1 - S);
				dU[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] * (1 + S*cos(H_g) / cos(pi / 3 - H_g));
				dU[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a + output_H_ * output_W_] * (1 + S - S*cos(H_g) / cos(pi / 3 - H_g));

			}

			else if ((H_g >= (2 * pi / 3)) && (H_g < (4 * pi / 3)))
			{
				H_g = H_g - 2 * pi / 3;

				//dU[i*output_H_ * output_W_*C + a] += -dV[i*output_H_ * output_W_*C + a + output_H_ * output_W_] * I_new*S*sin(pi / 3) / pow(cos(pi / 3 - H_g), 2);
				//dU[i*output_H_ * output_W_*C + a] += dV[i*output_H_ * output_W_*C + a] * I_new*S*sin(pi / 3) / pow(cos(pi / 3 - H_g), 2);

				//dU[i*output_H_ * output_W_*C + a + output_H_ * output_W_] += -dV[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] * I_new;
				//dU[i*output_H_ * output_W_*C + a + output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a + output_H_ * output_W_] * I_new*cos(H_g) / cos(pi / 3 - H_g);
				//dU[i*output_H_ * output_W_*C + a + output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a] * I_new*(1 - cos(H_g) / cos(pi / 3 - H_g));

				dU[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] * (1 - S);
				dU[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a + output_H_ * output_W_] * (1 + S*cos(H_g) / cos(pi / 3 - H_g));
				dU[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a] * (1 + S - S*cos(H_g) / cos(pi / 3 - H_g));

				  
			}

			else if ((H_g >= (4 * pi / 3)) && (H_g < (2 * pi)))
			{
				H_g = H_g - 4 * pi / 3;

				//dU[i*output_H_ * output_W_*C + a] += -dV[i*output_H_ * output_W_*C + a] * I_new*S*sin(pi / 3) / pow(cos(pi / 3 - H_g), 2);
				//dU[i*output_H_ * output_W_*C + a] += dV[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] * I_new*S*sin(pi / 3) / pow(cos(pi / 3 - H_g), 2);

				//dU[i*output_H_ * output_W_*C + a + output_H_ * output_W_] += -dV[i*output_H_ * output_W_*C + a + output_H_ * output_W_] * I_new;
				//dU[i*output_H_ * output_W_*C + a + output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a] * I_new*cos(H_g) / cos(pi / 3 - H_g);
				//dU[i*output_H_ * output_W_*C + a + output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] * I_new*(1 - cos(H_g) / cos(pi / 3 - H_g));

				dU[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a + output_H_ * output_W_] * (1 - S);
				dU[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a] * (1 + S*cos(H_g) / cos(pi / 3 - H_g));
				dU[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] += dV[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] * (1 + S - S*cos(H_g) / cos(pi / 3 - H_g));

			}
		}
	}
	
	
 // }
}

#ifdef CPU_ONLY
STUB_GPU(HSI2RGBLayer);
#endif

INSTANTIATE_CLASS(HSI2RGBLayer);
REGISTER_LAYER_CLASS(HSI2RGB);



}  // namespace caffe
