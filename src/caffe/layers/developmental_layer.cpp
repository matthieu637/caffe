// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>
#include <boost/random.hpp>

#include "caffe/filler.hpp"
#include "caffe/layers/developmental_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DevelopmentalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  const DevelopmentalParameter& param = this->layer_param_.developmental_param();
  this->do_scale_ = param.scale();
  this->probabilistic_ = param.probabilist();
  const int num_output = bottom[0]->count(1);
  
  CHECK_LE(param.control_size(), num_output);
  if(param.control_size() != 0) {
    this->control_.resize(param.control_size());
    for(uint i=0;i<this->control_.size();i++)
      this->control_[i] = param.control(i);
  } else {
    this->control_.resize(num_output);
    for(uint i=0;i<this->control_.size();i++)
      this->control_[i] = i;
  }
  
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    vector<int> weight_shape(2);
    weight_shape[0] = 1;
    weight_shape[1] = this->control_.size();
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    
    FillerParameter uf;
    uf.set_min(0);
    uf.set_max(1);
    uf.set_type("uniform");
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(uf));
    weight_filler->Fill(this->blobs_[0].get());
  } // parameter initialization
  
  if(param.diff_compute())
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void DevelopmentalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  if(this->probabilistic_ <= 2)
    rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void DevelopmentalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int num_output = bottom[0]->count(1);
  const int batch = count/num_output;
//   std::cout << " forward " <<std::endl;
  if (this->phase_ == TRAIN) {
    // Create random numbers
    Dtype* proba = this->blobs_[0]->mutable_cpu_data();
    
    if(this->probabilistic_ <= 1){
      for(uint i=0;i<this->blobs_[0]->count();i++)
        if(proba[i] < 0.)
          proba[i] = 0.;
        else if(proba[i] > 1.)
          proba[i] = 1.;
    } else // 2 and 3 use fabs
      for(uint i=0;i<this->blobs_[0]->count();i++)
        if(proba[i] < 0.)
          proba[i] = 0.;
    
    unsigned int* mask;
    if(this->probabilistic_ <= 2){
      mask = rand_vec_.mutable_cpu_data();
      caffe_set(count, (uint) 1, mask);
    }

    //control size == proba size != count == mask size != num_output
    const uint* c = this->control_.data(); 
    if(this->probabilistic_ == 0)
      for(int j=0;j < batch;++j)
        caffe_rng_bernoulli((int)this->control_.size(), proba, mask, c, j*num_output);
    else if(this->probabilistic_ == 1) {
      for(int j=0;j < batch;++j)
        for (int i = 0; i < this->control_.size(); ++i)
          mask[j*num_output+c[i]] = proba[i] >= 0.5; 
        //0.5 unless probabilistic_ == 1
        //do nothing because proba[i] is in [0;1]
    } else if(this->probabilistic_ == 2) {
      for(int j=0;j < batch;++j)
        for (int i = 0; i < this->control_.size(); ++i){
          uint index = j*num_output+c[i];
          mask[index] = std::fabs(proba[i]) >= std::fabs(bottom_data[index]);
        }
    }
    
    if(this->probabilistic_ <= 2){
      uint y=0;
      for (int i = 0; i < count; ++i) {
        Dtype scale_ = 1.;
        if(this->do_scale_ && c[y] == (i%num_output) && proba[y] != 0.){
          scale_ = 1. / proba[y++];
          if(y >= this->control_.size())
            y=0;
        }
        top_data[i] = bottom_data[i] * mask[i] * scale_;
      } 
    } else {
      for(int j=0;j < batch;++j)
        for (int i = 0; i < this->control_.size(); ++i){
          uint index = j*num_output+c[i];
          if(std::fabs(proba[i]) < std::fabs(bottom_data[index]) || proba[i] == 0.)
            top_data[index] = proba[i];
          else
            top_data[index] = bottom_data[i];
        }
    }
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void DevelopmentalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  if (this->param_propagate_down_.size() > 0 && this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
//     caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
//                           N_, K_, M_,
//                           (Dtype)1., top_diff, bottom_data,
//                           (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    Dtype* param_diff = this->blobs_[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const int num_output = bottom[0]->count(1);
    const int batch = count/num_output;
    
    for (int i = 0; i < this->control_.size(); ++i) {
      double sum = 0.f;
      for(int j=0;j < batch;++j){
        uint index = j*num_output+this->control_[i];
        sum += top_diff[index] * bottom_data[index];
      }
      
      param_diff[i] = sum;
    }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask;
      if(this->probabilistic_ <= 2)
        mask = rand_vec_.cpu_data();
      const Dtype* proba = this->blobs_[0]->cpu_data();
      const uint* c = this->control_.data();
      const int count = bottom[0]->count();
      const int num_output = bottom[0]->count(1);
      uint y=0;
      if(this->probabilistic_ <=2)
        for (int i = 0; i < count; ++i) {
          Dtype scale_ = 1.;
          if(this->do_scale_ && c[y] == (i%num_output) && proba[y] != 0.){
            scale_ = 1. / proba[y++];
            if(y >= this->control_.size())
              y=0;
          }
          bottom_diff[i] = top_diff[i] * mask[i] * scale_;
        }
      else{
        const Dtype* bottom_data = bottom[0]->cpu_data();
        for (int i = 0; i < count; ++i) {
          if(c[y] == (i%num_output) && proba[y] != 0. && std::fabs(proba[y]) < std::fabs(bottom_data[i])){
            bottom_diff[i] = 0.;
            if(y >= this->control_.size())
              y=0;
          } else 
            bottom_diff[i] = top_diff[i];
        }
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DevelopmentalLayer);
#endif

INSTANTIATE_CLASS(DevelopmentalLayer);
REGISTER_LAYER_CLASS(Developmental);

}  // namespace caffe
