#include "../../include/variableContainer.h"

template <int dim, int degree, typename T>
variableContainer<dim, degree, T>::variableContainer(
  const dealii::MatrixFree<dim, double> &data,
  const std::vector<variable_info>      &_varInfoList,
  const std::vector<variable_info>      &_varChangeInfoList,
  const std::vector<variable_info>      &_varOldInfoList)
  : varInfoList(_varInfoList)
  , varChangeInfoList(_varChangeInfoList)
  , varOldInfoList(_varOldInfoList)
  , num_var(varInfoList.size())
{
  for (unsigned int i = 0; i < num_var; i++)
    {
      const auto &var_info        = varInfoList[i];
      const auto &var_change_info = varChangeInfoList[i];
      const auto &var_old_info    = varOldInfoList[i];

      if (var_info.var_needed)
        {
          const unsigned int var_index = var_info.global_var_index;

          if (var_info.is_scalar)
            {
              scalar_vars_map.emplace(var_index,
                                      std::make_unique<scalar_FEEval>(data, i));
            }
          else
            {
              vector_vars_map.emplace(var_index,
                                      std::make_unique<vector_FEEval>(data, i));
            }
        }

      if (var_change_info.var_needed)
        {
          const unsigned int var_index = var_change_info.global_var_index;

          if (var_change_info.is_scalar)
            {
              scalar_change_in_vars_map.emplace(var_index,
                                                std::make_unique<scalar_FEEval>(data, i));
            }
          else
            {
              vector_change_in_vars_map.emplace(var_index,
                                                std::make_unique<vector_FEEval>(data, i));
            }
        }

      if (var_old_info.var_needed)
        {
          const unsigned int var_index = var_old_info.global_var_index;

          if (var_old_info.is_scalar)
            {
              scalar_old_vars_map.emplace(var_index,
                                          std::make_unique<scalar_FEEval>(data, i));
            }
          else
            {
              vector_old_vars_map.emplace(var_index,
                                          std::make_unique<vector_FEEval>(data, i));
            }
        }
    }
}

template <int dim, int degree, typename T>
variableContainer<dim, degree, T>::variableContainer(
  const dealii::MatrixFree<dim, double> &data,
  const std::vector<variable_info>      &_varInfoList,
  const std::vector<variable_info>      &_varOldInfoList)
  : varInfoList(_varInfoList)
  , varOldInfoList(_varOldInfoList)
  , num_var(varInfoList.size())
{
  for (unsigned int i = 0; i < num_var; i++)
    {
      const auto &var_info     = varInfoList[i];
      const auto &var_old_info = varOldInfoList[i];

      if (var_info.var_needed)
        {
          const unsigned int var_index = var_info.global_var_index;

          if (var_info.is_scalar)
            {
              scalar_vars_map.emplace(var_index,
                                      std::make_unique<scalar_FEEval>(data, i));
            }
          else
            {
              vector_vars_map.emplace(var_index,
                                      std::make_unique<vector_FEEval>(data, i));
            }
        }

      if (var_old_info.var_needed)
        {
          const unsigned int var_index = var_old_info.global_var_index;

          if (var_old_info.is_scalar)
            {
              scalar_old_vars_map.emplace(var_index,
                                          std::make_unique<scalar_FEEval>(data, i));
            }
          else
            {
              vector_old_vars_map.emplace(var_index,
                                          std::make_unique<vector_FEEval>(data, i));
            }
        }
    }
}

template <int dim, int degree, typename T>
variableContainer<dim, degree, T>::variableContainer(
  const dealii::MatrixFree<dim, double> &data,
  const std::vector<variable_info>      &_varInfoList)
  : varInfoList(_varInfoList)
  , num_var(varInfoList.size())
{
  for (unsigned int i = 0; i < num_var; i++)
    {
      const auto &var_info = varInfoList[i];

      if (var_info.var_needed)
        {
          const unsigned int var_index = var_info.global_var_index;

          if (var_info.is_scalar)
            {
              scalar_vars_map.emplace(var_index,
                                      std::make_unique<scalar_FEEval>(data, i));
            }
          else
            {
              vector_vars_map.emplace(var_index,
                                      std::make_unique<vector_FEEval>(data, i));
            }
        }
    }
}

// Variant of the constructor where it reads from a fixed index of "data", used
// for post-processing
template <int dim, int degree, typename T>
variableContainer<dim, degree, T>::variableContainer(
  const dealii::MatrixFree<dim, double> &data,
  const std::vector<variable_info>      &_varInfoList,
  const unsigned int                    &fixed_index)
  : varInfoList(_varInfoList)
  , num_var(varInfoList.size())
{
  for (unsigned int i = 0; i < num_var; i++)
    {
      const auto &var_info = varInfoList[i];

      if (!var_info.var_needed)
        {
          continue;
        }

      const unsigned int var_index = var_info.global_var_index;

      if (var_info.is_scalar)
        {
          scalar_vars_map.emplace(var_index,
                                  std::make_unique<scalar_FEEval>(data, fixed_index));
        }
      else
        {
          vector_vars_map.emplace(var_index,
                                  std::make_unique<vector_FEEval>(data, fixed_index));
        }
    }
}

template <int dim, int degree, typename T>
unsigned int
variableContainer<dim, degree, T>::get_num_q_points() const
{
  if (!scalar_vars_map.empty())
    {
      return scalar_vars_map.begin()->second->n_q_points;
    }
  else if (!vector_vars_map.empty())
    {
      return vector_vars_map.begin()->second->n_q_points;
    }
  else if (!scalar_change_in_vars_map.empty())
    {
      return scalar_change_in_vars_map.begin()->second->n_q_points;
    }
  else if (!vector_change_in_vars_map.empty())
    {
      return vector_change_in_vars_map.begin()->second->n_q_points;
    }
  else
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "PRISMS-PF Error: When trying to access the number of quadrature "
                    "points, all FEEvaluation object containers were empty."));
    }
}

template <int dim, int degree, typename T>
dealii::Point<dim, T>
variableContainer<dim, degree, T>::get_q_point_location() const
{
  if (!scalar_vars_map.empty())
    {
      return scalar_vars_map.begin()->second->quadrature_point(q_point);
    }
  else if (!vector_vars_map.empty())
    {
      return vector_vars_map.begin()->second->quadrature_point(q_point);
    }
  else if (!scalar_change_in_vars_map.empty())
    {
      return scalar_change_in_vars_map.begin()->second->quadrature_point(q_point);
    }
  else if (!vector_change_in_vars_map.empty())
    {
      return vector_change_in_vars_map.begin()->second->quadrature_point(q_point);
    }
  else
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "PRISMS-PF Error: When trying to access the quadrature point "
                    "location, all FEEvaluation object containers were empty."));
    }
}

template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::reinit_and_eval(const std::vector<vectorType *> &src,
                                                   unsigned int                     cell)
{
  for (unsigned int i = 0; i < num_var; i++)
    {
      const auto &var_info = varInfoList[i];

      if (!var_info.var_needed)
        {
          continue;
        }

      const unsigned int var_index = var_info.global_var_index;

      if (var_info.is_scalar)
        {
          Assert(scalar_vars_map.find(var_index) != scalar_vars_map.end(),
                 dealii::ExcMessage(
                   "PRISMS-PF Error: Invalid index for accessing scalar_vars_map."));

          auto *scalar_FEEval_ptr = scalar_vars_map[var_index].get();
          scalar_FEEval_ptr->reinit(cell);
          scalar_FEEval_ptr->read_dof_values(*src[i]);
          scalar_FEEval_ptr->evaluate(var_info.evaluation_flags);
        }
      else
        {
          Assert(vector_vars_map.find(var_index) != vector_vars_map.end(),
                 dealii::ExcMessage(
                   "PRISMS-PF Error: Invalid index for accessing vector_vars_map."));

          auto *vector_FEEval_ptr = vector_vars_map[var_index].get();
          vector_FEEval_ptr->reinit(cell);
          vector_FEEval_ptr->read_dof_values(*src[i]);
          vector_FEEval_ptr->evaluate(var_info.evaluation_flags);
        }
    }
}

/**
 * This is specialized for the LHS where a change in solution is needed. The RHS method
 * takes the src as a vector of vectorTypes.
 */
template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::reinit_and_eval_change_in_solution(
  const vectorType &src,
  unsigned int      cell,
  unsigned int      var_being_solved)
{
  if (varChangeInfoList[var_being_solved].is_scalar)
    {
      Assert(
        scalar_change_in_vars_map.find(var_being_solved) !=
          scalar_change_in_vars_map.end(),
        dealii::ExcMessage(
          "PRISMS-PF Error: Invalid index for accessing scalar_change_in_vars_map."));

      auto *scalar_FEEval_ptr = scalar_change_in_vars_map[var_being_solved].get();
      scalar_FEEval_ptr->reinit(cell);
      scalar_FEEval_ptr->read_dof_values(src);
      scalar_FEEval_ptr->evaluate(varChangeInfoList[var_being_solved].evaluation_flags);
    }
  else
    {
      Assert(
        vector_change_in_vars_map.find(var_being_solved) !=
          vector_change_in_vars_map.end(),
        dealii::ExcMessage(
          "PRISMS-PF Error: Invalid index for accessing vector_change_in_vars_map."));

      auto *vector_FEEval_ptr = vector_change_in_vars_map[var_being_solved].get();
      vector_FEEval_ptr->reinit(cell);
      vector_FEEval_ptr->read_dof_values(src);
      vector_FEEval_ptr->evaluate(varChangeInfoList[var_being_solved].evaluation_flags);
    }
}

template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::reinit_and_eval_old_solution(
  const boost::unordered_map<unsigned int, std::unique_ptr<vectorType>> &src,
  unsigned int                                                           cell)
{
  for (const auto &old_solution : src)
    {
      const unsigned int field_index = old_solution.first;
      const auto        &field_info  = varOldInfoList[field_index];

      if (!field_info.var_needed)
        {
          continue;
        }

      const std::unique_ptr<vectorType> &old_solution_field = old_solution.second;

      if (field_info.is_scalar)
        {
          Assert(scalar_old_vars_map.find(field_index) != scalar_old_vars_map.end(),
                 dealii::ExcMessage(
                   "PRISMS-PF Error: Invalid index for accessing scalar_old_vars_map."));

          auto *scalar_FEEval_ptr = scalar_old_vars_map[field_index].get();
          scalar_FEEval_ptr->reinit(cell);
          scalar_FEEval_ptr->read_dof_values(*old_solution_field);
          scalar_FEEval_ptr->evaluate(field_info.evaluation_flags);
        }
      else
        {
          Assert(vector_old_vars_map.find(field_index) != vector_old_vars_map.end(),
                 dealii::ExcMessage(
                   "PRISMS-PF Error: Invalid index for accessing vector_old_vars_map."));

          auto *vector_FEEval_ptr = vector_old_vars_map[field_index].get();
          vector_FEEval_ptr->reinit(cell);
          vector_FEEval_ptr->read_dof_values(*old_solution_field);
          vector_FEEval_ptr->evaluate(field_info.evaluation_flags);
        }
    }
}

template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::reinit(unsigned int cell)
{
  for (unsigned int i = 0; i < num_var; i++)
    {
      const auto &var_info = varInfoList[i];

      if (!var_info.var_needed)
        {
          continue;
        }

      const unsigned int var_index = var_info.global_var_index;

      if (var_info.is_scalar)
        {
          Assert(scalar_vars_map.find(var_index) != scalar_vars_map.end(),
                 dealii::ExcMessage(
                   "PRISMS-PF Error: Invalid index for accessing scalar_vars_map."));

          scalar_vars_map[var_index]->reinit(cell);
        }
      else
        {
          Assert(vector_vars_map.find(var_index) != vector_vars_map.end(),
                 dealii::ExcMessage(
                   "PRISMS-PF Error: Invalid index for accessing vector_vars_map."));

          vector_vars_map[var_index]->reinit(cell);
        }
    }
}

template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::integrate_and_distribute(
  std::vector<vectorType *> &dst)
{
  for (unsigned int i = 0; i < num_var; i++)
    {
      const auto &var_info = varInfoList[i];

      if (var_info.residual_flags & dealii::EvaluationFlags::nothing)
        {
          continue;
        }

      const unsigned int var_index = var_info.global_var_index;

      if (var_info.is_scalar)
        {
          Assert(scalar_vars_map.find(var_index) != scalar_vars_map.end(),
                 dealii::ExcMessage(
                   "PRISMS-PF Error: Invalid index for accessing scalar_vars_map."));

          auto *scalar_FEEval_ptr = scalar_vars_map[var_index].get();
          scalar_FEEval_ptr->integrate(var_info.residual_flags);
          scalar_FEEval_ptr->distribute_local_to_global(*dst[i]);
        }
      else
        {
          Assert(vector_vars_map.find(var_index) != vector_vars_map.end(),
                 dealii::ExcMessage(
                   "PRISMS-PF Error: Invalid index for accessing vector_vars_map."));

          auto *vector_FEEval_ptr = vector_vars_map[var_index].get();
          vector_FEEval_ptr->integrate(var_info.residual_flags);
          vector_FEEval_ptr->distribute_local_to_global(*dst[i]);
        }
    }
}

template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::integrate_and_distribute_change_in_solution_LHS(
  vectorType        &dst,
  const unsigned int var_being_solved)
{
  // integrate
  if (varChangeInfoList[var_being_solved].is_scalar)
    {
      Assert(
        scalar_change_in_vars_map.find(var_being_solved) !=
          scalar_change_in_vars_map.end(),
        dealii::ExcMessage(
          "PRISMS-PF Error: Invalid index for accessing scalar_change_in_vars_map."));

      auto *scalar_FEEval_ptr = scalar_change_in_vars_map[var_being_solved].get();
      scalar_FEEval_ptr->integrate(varChangeInfoList[var_being_solved].residual_flags);
      scalar_FEEval_ptr->distribute_local_to_global(dst);
    }
  else
    {
      Assert(
        vector_change_in_vars_map.find(var_being_solved) !=
          vector_change_in_vars_map.end(),
        dealii::ExcMessage(
          "PRISMS-PF Error: Invalid index for accessing vector_change_in_vars_map."));

      auto *vector_FEEval_ptr = vector_change_in_vars_map[var_being_solved].get();
      vector_FEEval_ptr->integrate(varChangeInfoList[var_being_solved].residual_flags);
      vector_FEEval_ptr->distribute_local_to_global(dst);
    }
}

template <int dim, int degree, typename T>
T
variableContainer<dim, degree, T>::get_scalar_value(
  unsigned int global_variable_index) const
{
  Assert(varInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::values,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a scalar variable value "
                       "that was not marked as needed in 'equations.cc'. The attempted "
                       "access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return scalar_vars_map.at(global_variable_index)->get_value(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<1, dim, T>
variableContainer<dim, degree, T>::get_scalar_gradient(
  unsigned int global_variable_index) const
{
  Assert(varInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::gradients,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a scalar variable gradient "
                       "that was not marked as needed in 'equations.cc'. The attempted "
                       "access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return scalar_vars_map.at(global_variable_index)->get_gradient(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<2, dim, T>
variableContainer<dim, degree, T>::get_scalar_hessian(
  unsigned int global_variable_index) const
{
  Assert(varInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::hessians,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a scalar variable hessian "
                       "that was not marked as needed in 'equations.cc'. The attempted "
                       "access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return scalar_vars_map.at(global_variable_index)->get_hessian(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<1, dim, T>
variableContainer<dim, degree, T>::get_vector_value(
  unsigned int global_variable_index) const
{
  Assert(varInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::values,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a vector variable value "
                       "that was not marked as needed in 'equations.cc'. The attempted "
                       "access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return vector_vars_map.at(global_variable_index)->get_value(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<2, dim, T>
variableContainer<dim, degree, T>::get_vector_gradient(
  unsigned int global_variable_index) const
{
  Assert(varInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::gradients,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a vector variable gradient "
                       "that was not marked as needed in 'equations.cc'. The attempted "
                       "access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return vector_vars_map.at(global_variable_index)->get_gradient(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<3, dim, T>
variableContainer<dim, degree, T>::get_vector_hessian(
  unsigned int global_variable_index) const
{
  Assert(varInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::hessians,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a vector variable hessian "
                       "that was not marked as needed in 'equations.cc'. The attempted "
                       "access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return vector_vars_map.at(global_variable_index)->get_hessian(q_point);
}

template <int dim, int degree, typename T>
T
variableContainer<dim, degree, T>::get_change_in_scalar_value(
  unsigned int global_variable_index) const
{
  Assert(varChangeInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::values,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a change in scalar variable "
                       "value that was not marked as needed in 'equations.cc'. The "
                       "attempted access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return scalar_change_in_vars_map.at(global_variable_index)->get_value(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<1, dim, T>
variableContainer<dim, degree, T>::get_change_in_scalar_gradient(
  unsigned int global_variable_index) const
{
  Assert(varChangeInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::gradients,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a change in scalar variable "
                       "gradient that was not marked as needed in 'equations.cc'. The "
                       "attempted access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return scalar_change_in_vars_map.at(global_variable_index)->get_gradient(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<2, dim, T>
variableContainer<dim, degree, T>::get_change_in_scalar_hessian(
  unsigned int global_variable_index) const
{
  Assert(varChangeInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::hessians,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a change in scalar variable "
                       "hessian that was not marked as needed in 'equations.cc'. The "
                       "attempted access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return scalar_change_in_vars_map.at(global_variable_index)->get_hessian(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<1, dim, T>
variableContainer<dim, degree, T>::get_change_in_vector_value(
  unsigned int global_variable_index) const
{
  Assert(varChangeInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::values,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a change in vector variable "
                       "value that was not marked as needed in 'equations.cc'. The "
                       "attempted access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return vector_change_in_vars_map.at(global_variable_index)->get_value(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<2, dim, T>
variableContainer<dim, degree, T>::get_change_in_vector_gradient(
  unsigned int global_variable_index) const
{
  Assert(varChangeInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::gradients,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a change in vector variable "
                       "gradient that was not marked as needed in 'equations.cc'. The "
                       "attempted access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return vector_change_in_vars_map.at(global_variable_index)->get_gradient(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<3, dim, T>
variableContainer<dim, degree, T>::get_change_in_vector_hessian(
  unsigned int global_variable_index) const
{
  Assert(varChangeInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::hessians,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a change in vector variable "
                       "hessian that was not marked as needed in 'equations.cc'. The "
                       "attempted access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return vector_change_in_vars_map.at(global_variable_index)->get_hessian(q_point);
}

template <int dim, int degree, typename T>
T
variableContainer<dim, degree, T>::get_old_scalar_value(
  unsigned int global_variable_index) const
{
  Assert(varOldInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::values,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a previous scalar variable "
                       "value that was not marked as needed in 'equations.cc'. The "
                       "attempted access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return scalar_old_vars_map.at(global_variable_index)->get_value(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<1, dim, T>
variableContainer<dim, degree, T>::get_old_scalar_gradient(
  unsigned int global_variable_index) const
{
  Assert(varOldInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::gradients,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a previous scalar variable "
                       "gradient that was not marked as needed in 'equations.cc'. The "
                       "attempted access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return scalar_old_vars_map.at(global_variable_index)->get_gradient(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<2, dim, T>
variableContainer<dim, degree, T>::get_old_scalar_hessian(
  unsigned int global_variable_index) const
{
  Assert(varOldInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::hessians,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a previous scalar variable "
                       "hessian that was not marked as needed in 'equations.cc'. The "
                       "attempted access was for variable with index ") +
           std::to_string(global_variable_index) + "."));

  return scalar_old_vars_map.at(global_variable_index)->get_hessian(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<1, dim, T>
variableContainer<dim, degree, T>::get_old_vector_value(
  unsigned int global_variable_index) const
{
  Assert(varOldInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::values,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a previous vector variable "
                       "value that was not marked as needed in 'equations.cc'. The "
                       "attempted access was for variable with index ") +
           std::to_string(global_variable_index) + " ."));

  return vector_old_vars_map.at(global_variable_index)->get_value(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<2, dim, T>
variableContainer<dim, degree, T>::get_old_vector_gradient(
  unsigned int global_variable_index) const
{
  Assert(varOldInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::gradients,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a previous vector variable "
                       "gradient that was not marked as needed in 'equations.cc'. The "
                       "attempted access was for variable with index ") +
           std::to_string(global_variable_index) + " ."));

  return vector_old_vars_map.at(global_variable_index)->get_gradient(q_point);
}

template <int dim, int degree, typename T>
dealii::Tensor<3, dim, T>
variableContainer<dim, degree, T>::get_old_vector_hessian(
  unsigned int global_variable_index) const
{
  Assert(varOldInfoList[global_variable_index].evaluation_flags &
           dealii::EvaluationFlags::hessians,
         dealii::ExcMessage(
           std::string("PRISMS-PF Error: Attempted access of a previous vector variable "
                       "hessian that was not marked as needed in 'equations.cc'. The "
                       "attempted access was for variable with index ") +
           std::to_string(global_variable_index) + " ."));

  return vector_old_vars_map.at(global_variable_index)->get_hessian(q_point);
}

// The methods to set the residual terms
template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::set_scalar_value_term_RHS(
  unsigned int global_variable_index,
  T            val)
{
  scalar_vars_map[global_variable_index]->submit_value(val, q_point);
}

template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::set_scalar_gradient_term_RHS(
  unsigned int              global_variable_index,
  dealii::Tensor<1, dim, T> grad)
{
  scalar_vars_map[global_variable_index]->submit_gradient(grad, q_point);
}

template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::set_vector_value_term_RHS(
  unsigned int              global_variable_index,
  dealii::Tensor<1, dim, T> val)
{
  vector_vars_map[global_variable_index]->submit_value(val, q_point);
}

template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::set_vector_gradient_term_RHS(
  unsigned int              global_variable_index,
  dealii::Tensor<2, dim, T> grad)
{
  vector_vars_map[global_variable_index]->submit_gradient(grad, q_point);
}

template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::set_scalar_value_term_LHS(
  unsigned int global_variable_index,
  T            val)
{
  scalar_change_in_vars_map[global_variable_index]->submit_value(val, q_point);
}

template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::set_scalar_gradient_term_LHS(
  unsigned int              global_variable_index,
  dealii::Tensor<1, dim, T> grad)
{
  scalar_change_in_vars_map[global_variable_index]->submit_gradient(grad, q_point);
}

template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::set_vector_value_term_LHS(
  unsigned int              global_variable_index,
  dealii::Tensor<1, dim, T> val)
{
  vector_change_in_vars_map[global_variable_index]->submit_value(val, q_point);
}

template <int dim, int degree, typename T>
void
variableContainer<dim, degree, T>::set_vector_gradient_term_LHS(
  unsigned int              global_variable_index,
  dealii::Tensor<2, dim, T> grad)
{
  vector_change_in_vars_map[global_variable_index]->submit_gradient(grad, q_point);
}

template class variableContainer<2, 1, dealii::VectorizedArray<double>>;
template class variableContainer<3, 1, dealii::VectorizedArray<double>>;

template class variableContainer<2, 2, dealii::VectorizedArray<double>>;
template class variableContainer<3, 2, dealii::VectorizedArray<double>>;

template class variableContainer<2, 3, dealii::VectorizedArray<double>>;
template class variableContainer<3, 3, dealii::VectorizedArray<double>>;

template class variableContainer<2, 4, dealii::VectorizedArray<double>>;
template class variableContainer<3, 4, dealii::VectorizedArray<double>>;

template class variableContainer<2, 5, dealii::VectorizedArray<double>>;
template class variableContainer<3, 5, dealii::VectorizedArray<double>>;

template class variableContainer<2, 6, dealii::VectorizedArray<double>>;
template class variableContainer<3, 6, dealii::VectorizedArray<double>>;