// Model Variables Class

#ifndef INCLUDE_MODELVARIABLE_H_
#define INCLUDE_MODELVARIABLE_H_

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/matrix_free/evaluation_flags.h>

template <int dim>
class modelVariable
{
public:
  dealii::VectorizedArray<double>                         scalarValue;
  dealii::Tensor<1, dim, dealii::VectorizedArray<double>> scalarGrad;
  dealii::Tensor<2, dim, dealii::VectorizedArray<double>> scalarHess;

  dealii::Tensor<1, dim, dealii::VectorizedArray<double>> vectorValue;
  dealii::Tensor<2, dim, dealii::VectorizedArray<double>> vectorGrad;
  dealii::Tensor<3, dim, dealii::VectorizedArray<double>> vectorHess;
};

template <int dim>
class modelResidual
{
public:
  dealii::VectorizedArray<double>                         scalarValueResidual;
  dealii::Tensor<1, dim, dealii::VectorizedArray<double>> scalarGradResidual;

  dealii::Tensor<1, dim, dealii::VectorizedArray<double>> vectorValueResidual;
  dealii::Tensor<2, dim, dealii::VectorizedArray<double>> vectorGradResidual;
};

struct variable_info
{
  // Whether field is scalar
  bool is_scalar;

  // Global index of the field
  unsigned int global_var_index;

  // Time index of the field. The default value is 0, which is the value at the current
  // index. To access the previous timestep, a value of -1 would be set.
  int time_index;

  // Evaluation flags for the field (value/gradient/hessian)
  dealii::EvaluationFlags::EvaluationFlags evaluation_flags;

  // Residual flags for the field (value/gradient). Currently, hessian residuals are
  // unsupported
  dealii::EvaluationFlags::EvaluationFlags residual_flags;

  // Whether the variable is needed
  bool var_needed;

  // Constructor with default values
  variable_info(
    bool                                     scalar   = true,
    unsigned int                             global_i = 0,
    int                                      time_i   = 0,
    dealii::EvaluationFlags::EvaluationFlags eval     = dealii::EvaluationFlags::nothing,
    dealii::EvaluationFlags::EvaluationFlags res      = dealii::EvaluationFlags::nothing,
    bool                                     need_var = false)
    : is_scalar(scalar)
    , global_var_index(global_i)
    , time_index(time_i)
    , evaluation_flags(eval)
    , residual_flags(res)
    , var_needed(need_var)
  {}
};

#endif /* INCLUDE_MODELVARIABLE_H_ */
