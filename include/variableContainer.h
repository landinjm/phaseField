// This class permits the access of a subset of indexed fields and gives an
// error if any non-allowed fields are requested
#ifndef VARIBLECONTAINER_H
#define VARIBLECONTAINER_H

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <boost/unordered_map.hpp>

#include "userInputParameters.h"

template <int dim, int degree, typename T>
class variableContainer
{
public:
#include "typeDefs.h"

  // Contructor of variable container for nonexplicit equations
  variableContainer(const dealii::MatrixFree<dim, double> &data,
                    const std::vector<variable_info>      &_varInfoList,
                    const std::vector<variable_info>      &_varChangeInfoList);

  // Contructor of variable container for explicit equations
  variableContainer(const dealii::MatrixFree<dim, double> &data,
                    const std::vector<variable_info>      &_varInfoList);

  // Contructor of variable container for postprocessing
  variableContainer(const dealii::MatrixFree<dim, double> &data,
                    const std::vector<variable_info>      &_varInfoList,
                    const unsigned int                    &fixed_index);

  // Return the value of the scalar field at this index
  T
  get_scalar_value(unsigned int global_variable_index, int time_index = 0) const;

  // Return the gradient of the scalar field at this index
  dealii::Tensor<1, dim, T>
  get_scalar_gradient(unsigned int global_variable_index, int time_index = 0) const;

  // Return the hessian of the scalar field at this index
  dealii::Tensor<2, dim, T>
  get_scalar_hessian(unsigned int global_variable_index, int time_index = 0) const;

  // Return the value of the vector field at this index
  dealii::Tensor<1, dim, T>
  get_vector_value(unsigned int global_variable_index, int time_index = 0) const;

  // Return the gradient of the vector field at this index
  dealii::Tensor<2, dim, T>
  get_vector_gradient(unsigned int global_variable_index, int time_index = 0) const;

  // Return the hessian of the vector field at this index
  dealii::Tensor<3, dim, T>
  get_vector_hessian(unsigned int global_variable_index, int time_index = 0) const;

  // Return the change in value of the scalar field at this index
  T
  get_change_in_scalar_value(unsigned int global_variable_index) const;

  // Return the change in gradient of the scalar field at this index
  dealii::Tensor<1, dim, T>
  get_change_in_scalar_gradient(unsigned int global_variable_index) const;

  // Return the change in hessian of the scalar field at this index
  dealii::Tensor<2, dim, T>
  get_change_in_scalar_hessian(unsigned int global_variable_index) const;

  // Return the change in value of the vector field at this index
  dealii::Tensor<1, dim, T>
  get_change_in_vector_value(unsigned int global_variable_index) const;

  // Return the change in gradient of the vector field at this index
  dealii::Tensor<2, dim, T>
  get_change_in_vector_gradient(unsigned int global_variable_index) const;

  // Return the change in hessian of the vector field at this index
  dealii::Tensor<3, dim, T>
  get_change_in_vector_hessian(unsigned int global_variable_index) const;

  // Set the RHS value residual term of the scalar field at this index
  void
  set_scalar_value_term_RHS(unsigned int global_variable_index,
                            T            val,
                            int          time_index = 0);

  // Set the RHS gradient residual term of the scalar field at this index
  void
  set_scalar_gradient_term_RHS(unsigned int              global_variable_index,
                               dealii::Tensor<1, dim, T> grad,
                               int                       time_index = 0);

  // Set the RHS value residual term of the vector field at this index
  void
  set_vector_value_term_RHS(unsigned int              global_variable_index,
                            dealii::Tensor<1, dim, T> val,
                            int                       time_index = 0);

  // Set the RHS gradient residual term of the vector field at this index
  void
  set_vector_gradient_term_RHS(unsigned int              global_variable_index,
                               dealii::Tensor<2, dim, T> grad,
                               int                       time_index = 0);

  // Set the LHS value residual term of the scalar field at this index
  void
  set_scalar_value_term_LHS(unsigned int global_variable_index, T val);

  // Set the LHS gradient residual term of the scalar field at this index
  void
  set_scalar_gradient_term_LHS(unsigned int              global_variable_index,
                               dealii::Tensor<1, dim, T> grad);

  // Set the LHS value residual term of the vector field at this index
  void
  set_vector_value_term_LHS(unsigned int              global_variable_index,
                            dealii::Tensor<1, dim, T> val);

  // Set the LHS gradient residual term of the vector field at this index
  void
  set_vector_gradient_term_LHS(unsigned int              global_variable_index,
                               dealii::Tensor<2, dim, T> grad);

  // Initialize, read DOFs, and set evaulation flags for each variable
  void
  reinit_and_eval(const std::vector<vectorType *> &src, unsigned int cell);

  // Initialize, read DOFs, and set evaulation flags for a specified variable
  void
  reinit_and_eval_change_in_solution(const vectorType &src,
                                     unsigned int      cell,
                                     unsigned int      var_being_solved);

  // Initialize the FEEvaluation object for each variable (used for post-processing)
  void
  reinit(unsigned int cell);

  // Integrate the residuals for all variables and distribute from local to global
  void
  integrate_and_distribute(std::vector<vectorType *> &dst);

  // Integrate the residuals for a specified variable and distribute from local to global
  void
  integrate_and_distribute_change_in_solution_LHS(vectorType        &dst,
                                                  const unsigned int var_being_solved);

  // Quadrature point index
  unsigned int q_point;

  // Return the number of quadrature points per cell
  unsigned int
  get_num_q_points() const;

  // Return the xyz coordinates of the quadrature point
  dealii::Point<dim, T>
  get_q_point_location() const;

private:
  using scalar_FEEval = dealii::FEEvaluation<dim, degree, degree + 1, 1, double>;
  using vector_FEEval = dealii::FEEvaluation<dim, degree, degree + 1, dim, double>;

  // Unordered map of FEEvaluation objects for each active scalar variable
  boost::unordered_map<std::pair<unsigned int, int>, std::unique_ptr<scalar_FEEval>>
    scalar_vars_map;

  // Unordered map of FEEvaluation objects for each active vector variable
  boost::unordered_map<std::pair<unsigned int, int>, std::unique_ptr<vector_FEEval>>
    vector_vars_map;

  // Unordered map of FEEvaluation objects for each active scalar variable where the
  // change in value is needed
  boost::unordered_map<unsigned int, std::unique_ptr<scalar_FEEval>>
    scalar_change_in_vars_map;

  // Unordered map of FEEvaluation objects for each active vector variable where the
  // change in value is needed
  boost::unordered_map<unsigned int, std::unique_ptr<vector_FEEval>>
    vector_change_in_vars_map;

  // Vector of struct containing relevant information at each variable (index, evaluation
  // flags, etc.)
  std::vector<variable_info> varInfoList;

  // Vector of struct containing relevant information at each variable where the change in
  // value is needed(index, evaluation flags, etc.)
  std::vector<variable_info> varChangeInfoList;

  // Number of variables
  unsigned int num_var;
};

#endif
