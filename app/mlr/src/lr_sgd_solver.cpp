// Author: Dai Wei (wdai@cs.cmu.edu)
// Date: 2014.10.04

#include "lr_sgd_solver.hpp"
#include "common.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <fstream>
#include <petuum_ps_common/include/petuum_ps.hpp>
#include <ml/include/ml.hpp>
#include <ml/util/fastapprox/fastapprox.hpp>
#include <utility>

std::ofstream lr_fout("/ebs/joao/lr_sgd_solver_log", std::fstream::out | std::fstream::app);

namespace mlr {

LRSGDSolver::LRSGDSolver(const LRSGDSolverConfig& config) :
  w_table_(config.w_table),
  w_cache_(config.feature_dim),
  //w_delta_(config.feature_dim),
  w_cache_old_(config.feature_dim),
  feature_dim_(config.feature_dim),
  w_table_num_cols_(config.w_table_num_cols), lambda_(config.lambda) {
    lr_fout << "feature_dim: " << feature_dim_ << std::endl;
    lr_fout << "w_table_num_cols_: " << w_table_num_cols_ << std::endl;
    if (config.sparse_data) {
      // Sparse feature, dense weight.
      FeatureDotProductFun_ = petuum::ml::SparseDenseFeatureDotProduct;
    } else {
      // Both weight and feature are dense
      FeatureDotProductFun_ = petuum::ml::DenseDenseFeatureDotProduct;
    }
  }

void LRSGDSolver::MiniBatchSGD(
    const std::vector<petuum::ml::AbstractFeature<float>*>& features,
      const std::vector<int32_t>& labels, const std::vector<int32_t>& idx,
      double lr) {
  petuum::HighResolutionTimer MiniBatchSGDTimer;
  lr_fout << "MiniBatchSGD: " << "\n";
  lr_fout << "lambda: " << lambda_ << "\n";
  lr_fout << "idx size: " << idx.size() << "\n";
  lr_fout << "lr: " << lr << "\n";
  if (lambda_ == 0) {
    for (const auto& i : idx) {
      petuum::ml::AbstractFeature<float>& feature = *features[i];
      //lr_fout << "feature num entries: " << feature.GetNumEntries() << std::endl;
      int32_t label = labels[i];
      CHECK_NE(-1, label) << "class label for binary logistic regression "
        << "should be {0,1}, not {-1,1}";
      const auto pred = Predict(feature);

      petuum::ml::FeatureScaleAndAdd(-lr * (pred[0] - label),
          feature, &w_cache_);
    }
  } else {
    petuum::ml::DenseFeature<float> w_delta(w_cache_.GetFeatureDim());
    for (const auto& i : idx) {
      petuum::ml::AbstractFeature<float>& feature = *features[i];
      //lr_fout << "feature num entries: " << feature.GetNumEntries() << std::endl;
      int32_t label = labels[i];
      const auto pred = Predict(feature);

      petuum::ml::FeatureScaleAndAdd(-lr * (pred[0] - label),
          feature, &w_delta);
    }
    // Decay weight.
    auto& w_cache_vec = w_cache_.GetVector();
    auto& w_delta_vec = w_delta.GetVector();
    for (int i = 0; i < w_cache_vec.size(); ++i) {
      w_cache_vec[i] = w_cache_vec[i] * (1-lr * lambda_)
        + w_delta_vec[i];
    }
  }
  lr_fout << std::fixed << "MiniBatchSGD elapsed: " << MiniBatchSGDTimer.elapsed() << "\n";
}

std::vector<float> LRSGDSolver::Predict(
    const petuum::ml::AbstractFeature<float>& feature) const {
  std::vector<float> y_vec(2);
  // fastsigmoid is numerically unstable for input <-88.
  y_vec[0] = petuum::ml::Sigmoid(FeatureDotProductFun_(feature, w_cache_));
  y_vec[1] = 1 - y_vec[0];
  return std::move(y_vec);
}

int32_t LRSGDSolver::ZeroOneLoss(const std::vector<float>& prediction, int32_t label)
  const {
    // prediction[0] is the probability of being in class 1
    return prediction[label] >= 0.5 ? 1 : 0;
  }

float LRSGDSolver::CrossEntropyLoss(const std::vector<float>& prediction, int32_t label)
  const {
    CHECK_LE(prediction[label], 1);
    CHECK_GE(prediction[label], 0);
    // prediction[0] is the probability of being in class 1
    float prob = (label == 0) ? prediction[1] : prediction[0];
    return -petuum::ml::SafeLog(prob);
  }

void LRSGDSolver::RefreshParams() {
  petuum::HighResolutionTimer refresh_timer;
  std::vector<float> w_delta(feature_dim_);
  std::vector<float>& w_cache_vec = w_cache_.GetVector();
  for (int i = 0; i < feature_dim_; ++i) {
    w_delta[i] = w_cache_vec[i] - w_cache_old_[i];
  }
  // Write delta's to PS table.
  int num_full_rows = feature_dim_ / w_table_num_cols_;
  for (int i = 0; i < num_full_rows; ++i) {
    petuum::DenseUpdateBatch<float> w_update_batch(0, w_table_num_cols_);
    for (int j = 0; j < w_table_num_cols_; ++j) {
      int idx = i * w_table_num_cols_ + j;
      CHECK_EQ(w_delta[idx], w_delta[idx]) << "nan detected.";
      w_update_batch[j] = w_delta[idx];
    }
    w_table_.DenseBatchInc(i, w_update_batch);
  }
  auto elapsed1 = refresh_timer.elapsed();

  // last incomplete row.
  if (feature_dim_ % w_table_num_cols_ != 0) {
    int num_cols_last_row = feature_dim_ - num_full_rows * w_table_num_cols_;
    petuum::DenseUpdateBatch<float> w_update_batch(0, num_cols_last_row);
    for (int j = 0; j < num_cols_last_row; ++j) {
      int idx = num_full_rows * w_table_num_cols_ + j;
      CHECK_EQ(w_delta[idx], w_delta[idx]) << "nan detected.";
      w_update_batch[j] = w_delta[idx];
    }
    w_table_.DenseBatchInc(num_full_rows, w_update_batch);
  }
  auto elapsed2 = refresh_timer.elapsed();

  // Read w from the PS.
  std::vector<float> w_cache(w_table_num_cols_);
  for (int i = 0; i < num_full_rows; ++i) {
    petuum::RowAccessor row_acc;
    const auto& r = w_table_.Get<petuum::DenseRow<float>>(i, &row_acc);
    r.CopyToVector(&w_cache);
    std::copy(w_cache.begin(), w_cache.end(),
              w_cache_vec.begin() + i * w_table_num_cols_);
  }
  auto elapsed3 = refresh_timer.elapsed();
  if (feature_dim_ % w_table_num_cols_ != 0) {
    // last incomplete row.
    int num_cols_last_row = feature_dim_ - num_full_rows * w_table_num_cols_;
    petuum::RowAccessor row_acc;
    const auto& r = w_table_.Get<petuum::DenseRow<float>>(num_full_rows,
        &row_acc);
    r.CopyToVector(&w_cache);
    std::copy(w_cache.begin(), w_cache.begin() + num_cols_last_row,
        w_cache_vec.begin() + num_full_rows * w_table_num_cols_);
  }
  w_cache_old_ = w_cache_vec;
  auto elapsed4 = refresh_timer.elapsed();

  static int counter = 0;
  if (counter++ % 100 == 0) {
	  lr_fout
		  << "elapsed1: " << elapsed1
		  << "elapsed2: " << elapsed2 - elapsed1
		  << "elapsed3: " << elapsed3 - elapsed2
		  << "elapsed4: " << elapsed4 - elapsed3
		  << "\n";
  }
}

void LRSGDSolver::SaveWeights(const std::string& filename) const {
  LOG(ERROR) << "SaveWeights is not implemented for binary LR.";
}

// 1/2 * lambda * ||w||^2
float LRSGDSolver::EvaluateL2RegLoss() const {
  double l2_norm = 0.;
  std::vector<float> w = w_cache_.GetVector();
  for (int i = 0; i < feature_dim_; ++i) {
    l2_norm += w[i] * w[i];
  }
  return 0.5 * lambda_ * l2_norm;
}

}  // namespace mlr
