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
  
  const int num_output = bottom[0]->count(1);
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>({1, num_output}));
    
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
  if (this->phase_ == TRAIN) {
    // Create random numbers
    const Dtype* proba = this->blobs_[0]->cpu_data();
    caffe_rng_bernoulli(count, proba, mask);
    for (int i = 0; i < count; ++i) {
      Dtype scale_ = 1. / proba[i];
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
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
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        Dtype scale_ = 1. / proba[i];
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
