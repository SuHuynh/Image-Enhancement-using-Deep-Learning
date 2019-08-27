#include <cmath>
#include <math.h>    
#include <vector>

#include "caffe/layers/rgb_conv_hsi.hpp"

const double pi = 3.141592654;

namespace caffe {


template <typename Dtype>
void RGBxxxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	N = bottom[0]->shape(0);
	C = bottom[0]->shape(1);
	H = bottom[0]->shape(2);
	W = bottom[0]->shape(3);
	output_H_ = bottom[0]->shape(2);
	output_W_ = bottom[0]->shape(3);
 }
	
	
template <typename Dtype>
void RGBxxxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* U = bottom[0]->cpu_data();
  Dtype* V = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < N; ++i) 
  {
	  for(int a=0; a < output_W_ * output_H_; ++a)
	  {
		Dtype B = U[i*output_H_ * output_W_*C + a];
		Dtype G = U[i*output_H_ * output_W_*C + a + output_H_ * output_W_] ;
		Dtype R = U[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_] ;
		 	 
		Dtype tu = 0.5* ((R-G)+(R-B);
		Dtype mau1 = pow(R-G,2);
		Dtype mau2 = R-B;
		Dtype mau3 = G-B;
		Dtype mau = pow(mau1 + mau2*mau3 , 0.5);

		Dtype goc = acos(tu / (mau + 0.00001));
		Dtype H_g ;

		if (B <= G)
		{
			H_g = goc;
		}
		else
		{
			H_g = 2 * pi - goc;
		}

			//H = H /(2*pi);

		Dtype minS = min_function(R,G,B);
		Dtype S = Dtype(1) - 3 * minS / (R+G+B+0.0000001);

		Dtype II = (R+G+B)/3;

		V[i*output_H_ * output_W_*C + a] = H_g;
		V[i*output_H_ * output_W_*C + a + output_H_ * output_W_]= S;
		V[i*output_H_ * output_W_*C + a + 2 * output_H_ * output_W_]= II;
		 		 	 	 		 	 
	  }
	 	  
   
  }
}

template <typename Dtype>
void RGBxxxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
			
 // }
}

#ifdef CPU_ONLY
STUB_GPU(RGBxxxLayer);
#endif

INSTANTIATE_CLASS(RGBxxxLayer);
REGISTER_LAYER_CLASS(RGBxxx);



}  // namespace caffe
