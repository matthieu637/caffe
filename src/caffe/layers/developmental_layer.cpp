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
  }  // parameter initialization
}

template <typename Dtype>
void DevelopmentalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void DevelopmentalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  const int num_output = bottom[0]->count(1);
  const int batch = count/num_output;
  if (this->phase_ == TRAIN) {
    caffe_set(count, (uint) 1, mask);
    // Create random numbers
    Dtype* proba = this->blobs_[0]->mutable_cpu_data();
    for(uint i=0;i<this->blobs_[0]->count();i++)
      if(proba[i] < 0.)
        proba[i] = 0.;
      else if(proba[i] > 1.)
        proba[i] = 1.;
    
    const uint* c = this->control_.data(); //control size == proba size != count == mask size 
    if(this->probabilistic_)
      for(int j=0;j < batch;++j)
        caffe_rng_bernoulli((int)this->control_.size(), proba, mask, c, j*num_output);
    else {
      for(int j=0;j < batch;++j)
        for (int i = 0; i < this->control_.size(); ++i)
          mask[j*num_output+c[i]] = proba[i] >= 0;
    }
    uint y=0;
    for (int i = 0; i < count; ++i) {
      Dtype scale_ = 1.;
      if(this->do_scale_ && c[y] == (i%batch)){
        scale_ = 1. / proba[y++];
        if(y >= num_output)
          y=0;
      }
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    } 
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
void DevelopmentalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const Dtype* proba = this->blobs_[0]->cpu_data();
      const uint* c = this->control_.data();
      const int count = bottom[0]->count();
      const int num_output = bottom[0]->count(1);
      const int batch = count/num_output;
      uint y=0;
      for (int i = 0; i < count; ++i) {
        Dtype scale_ = 1.;
        if(this->do_scale_ && c[y] == (i%batch)){
          scale_ = 1. / proba[y++];
          if(y >= num_output)
            y=0;
        }
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
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
