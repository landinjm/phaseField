#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <prismspf/core/exceptions.h>
#include <prismspf/core/matrix_free_operator.h>
#include <prismspf/core/pde_operator.h>
#include <prismspf/core/type_enums.h>
#include <prismspf/core/types.h>
#include <prismspf/core/variable_attributes.h>
#include <prismspf/core/variable_container.h>

#include <prismspf/config.h>

#include <map>
#include <memory>
#include <utility>
#include <vector>

PRISMS_PF_BEGIN_NAMESPACE

template <int dim, int degree, typename number>
matrixFreeOperator<dim, degree, number>::matrixFreeOperator(
  const std::map<unsigned int, variableAttributes>       &_attributes_list,
  std::shared_ptr<const PDEOperator<dim, degree, number>> _pde_operator,
  types::index                                            _current_index)
  : Subscriptor()
  , attributes_list(&_attributes_list)
  , pde_operator(std::move(_pde_operator))
  , current_index(_current_index)
{}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::initialize(
  std::shared_ptr<const dealii::MatrixFree<dim, number, size_type>> _data,
  const std::vector<unsigned int> &selected_field_indexes)
{
  data = std::move(_data);

  selected_fields.clear();
  if (selected_field_indexes.empty())
    {
      for (unsigned int i = 0; i < data->n_components(); ++i)
        {
          selected_fields.push_back(i);
        }
    }
  else
    {
      for (unsigned int i = 0; i < selected_field_indexes.size(); ++i)
        {
          AssertIndexRange(selected_field_indexes[i], data->n_components());
          for (unsigned int j = 0; j < selected_field_indexes.size(); ++j)
            {
              if (j != i)
                {
                  Assert(selected_field_indexes[j] != selected_field_indexes[i],
                         dealii::ExcMessage("Given row indices must be unique"));
                }
            }
          selected_fields.push_back(selected_field_indexes[i]);
        }
    }

  edge_constrained_indices.clear();
  edge_constrained_indices.resize(selected_fields.size());
}

template <int dim, int degree, typename number>
dealii::types::global_dof_index
matrixFreeOperator<dim, degree, number>::m() const
{
  Assert(data.get() != nullptr, dealii::ExcNotInitialized());

  const unsigned int total_size =
    std::accumulate(selected_fields.begin(),
                    selected_fields.end(),
                    0U,
                    [this](unsigned int sum, unsigned int field)
                    {
                      return sum + data->get_vector_partitioner(field)->size();
                    });

  return total_size;
}

template <int dim, int degree, typename number>
number
matrixFreeOperator<dim, degree, number>::el(
  [[maybe_unused]] const unsigned int &row,
  [[maybe_unused]] const unsigned int &col) const
{
  AssertThrow(false, FeatureNotImplemented("el()"));
  return 0.0;
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::clear()
{
  data.reset();
  inverse_diagonal_entries.reset();
  global_to_local_solution.clear();
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::initialize_dof_vector(
  VectorType  &dst,
  unsigned int dof_handler_index) const
{
  data->initialize_dof_vector(dst, dof_handler_index);
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::set_constrained_entries_to_one(
  VectorType &dst) const
{
  for (unsigned int j = 0; j < dealii::MatrixFreeOperators::BlockHelper::n_blocks(dst);
       ++j)
    {
      const std::vector<unsigned int> &constrained_dofs =
        data->get_constrained_dofs(selected_fields[j]);
      for (const auto constrained_dof : constrained_dofs)
        {
          dealii::MatrixFreeOperators::BlockHelper::subblock(dst, j).local_element(
            constrained_dof) = 1.0;
        }
      for (unsigned int i = 0; i < edge_constrained_indices[j].size(); ++i)
        {
          dealii::MatrixFreeOperators::BlockHelper::subblock(dst, j).local_element(
            edge_constrained_indices[j][i]) = 1.0;
        }
    }
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::add_global_to_local_mapping(
  const std::map<std::pair<unsigned int, dependencyType>, unsigned int>
    &_global_to_local_solution)
{
  global_to_local_solution = _global_to_local_solution;
}

template <int dim, int degree, typename number>
std::shared_ptr<const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>>>
matrixFreeOperator<dim, degree, number>::get_matrix_free() const
{
  return data;
}

template <int dim, int degree, typename number>
const std::shared_ptr<
  dealii::DiagonalMatrix<typename matrixFreeOperator<dim, degree, number>::VectorType>> &
matrixFreeOperator<dim, degree, number>::get_matrix_diagonal_inverse() const
{
  Assert(inverse_diagonal_entries.get() != nullptr && inverse_diagonal_entries->m() > 0,
         dealii::ExcNotInitialized());
  return inverse_diagonal_entries;
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::add_src_solution_subset(
  const std::vector<VectorType *> &_src_solution_subset)
{
  src_solution_subset = _src_solution_subset;
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::compute_explicit_update(
  std::vector<VectorType *>       &dst,
  const std::vector<VectorType *> &src) const
{
  Assert(!global_to_local_solution.empty(),
         dealii::ExcMessage(
           "The global to local solution mapping must not be empty. Make sure to call "
           "add_global_to_local_mapping() prior to any computations."));
  Assert(!dst.empty(), dealii::ExcMessage("The dst vector must not be empty"));
  Assert(!src.empty(), dealii::ExcMessage("The src vector must not be empty"));

  this->data->cell_loop(&matrixFreeOperator::compute_local_explicit_update,
                        this,
                        dst,
                        src,
                        true);
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::compute_postprocess_explicit_update(
  std::vector<VectorType *>       &dst,
  const std::vector<VectorType *> &src) const
{
  Assert(!global_to_local_solution.empty(),
         dealii::ExcMessage(
           "The global to local solution mapping must not be empty. Make sure to call "
           "add_global_to_local_mapping() prior to any computations."));
  Assert(!dst.empty(), dealii::ExcMessage("The dst vector must not be empty"));
  Assert(!src.empty(), dealii::ExcMessage("The src vector must not be empty"));

  this->data->cell_loop(&matrixFreeOperator::compute_local_postprocess_explicit_update,
                        this,
                        dst,
                        src,
                        true);
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::compute_nonexplicit_auxiliary_update(
  std::vector<VectorType *>       &dst,
  const std::vector<VectorType *> &src) const
{
  Assert(!global_to_local_solution.empty(),
         dealii::ExcMessage(
           "The global to local solution mapping must not be empty. Make sure to call "
           "add_global_to_local_mapping() prior to any computations."));
  Assert(!dst.empty(), dealii::ExcMessage("The dst vector must not be empty"));
  Assert(!src.empty(), dealii::ExcMessage("The src vector must not be empty"));

  this->data->cell_loop(&matrixFreeOperator::compute_local_nonexplicit_auxiliary_update,
                        this,
                        dst,
                        src,
                        true);
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::compute_residual(VectorType       &dst,
                                                          const VectorType &src) const
{
  Assert(!global_to_local_solution.empty(),
         dealii::ExcMessage(
           "The global to local solution mapping must not be empty. Make sure to call "
           "add_global_to_local_mapping() prior to any computations."));
  Assert(!src_solution_subset.empty(),
         dealii::ExcMessage("The src_solution_subset vector must not be empty"));
  Assert(dst.size() != 0,
         dealii::ExcMessage("The dst vector should not have size equal to 0"));
  Assert(src.size() != 0,
         dealii::ExcMessage("The src vector should not have size equal to 0"));

  this->data->cell_loop(&matrixFreeOperator::compute_local_residual,
                        this,
                        dst,
                        src,
                        true);
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::vmult(VectorType       &dst,
                                               const VectorType &src) const
{
  Assert(!global_to_local_solution.empty(),
         dealii::ExcMessage(
           "The global to local solution mapping must not be empty. Make sure to call "
           "add_global_to_local_mapping() prior to any computations."));
  Assert(dst.size() != 0,
         dealii::ExcMessage("The dst vector should not have size equal to 0"));
  Assert(src.size() != 0,
         dealii::ExcMessage("The src vector should not have size equal to 0"));

  this->data->cell_loop(&matrixFreeOperator::compute_local_newton_update,
                        this,
                        dst,
                        src,
                        true);
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::Tvmult(VectorType       &dst,
                                                const VectorType &src) const
{
  this->vmult(dst, src);
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::compute_local_explicit_update(
  const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &data,
  std::vector<VectorType *>                                              &dst,
  const std::vector<VectorType *>                                        &src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // Constructor for FEEvaluation objects
  variableContainer<dim, degree, number> variable_list(data,
                                                       *attributes_list,
                                                       global_to_local_solution,
                                                       solveType::EXPLICIT_RHS);

  // Initialize, evaluate, and submit based on user function.
  variable_list.eval_local_operator(
    [this](variableContainer<dim, degree, number> &var_list,
           const dealii::Point<dim, size_type>    &q_point_loc)
    {
      this->pde_operator->compute_explicit_RHS(var_list, q_point_loc);
    },
    dst,
    src,
    cell_range);
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::compute_local_postprocess_explicit_update(
  const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &data,
  std::vector<VectorType *>                                              &dst,
  const std::vector<VectorType *>                                        &src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // Constructor for FEEvaluation objects
  variableContainer<dim, degree, number> variable_list(data,
                                                       *attributes_list,
                                                       global_to_local_solution,
                                                       solveType::POSTPROCESS);

  // Initialize, evaluate, and submit based on user function.
  variable_list.eval_local_operator(
    [this](variableContainer<dim, degree, number> &var_list,
           const dealii::Point<dim, size_type>    &q_point_loc)
    {
      this->pde_operator->compute_postprocess_explicit_RHS(var_list, q_point_loc);
    },
    dst,
    src,
    cell_range);
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::compute_local_nonexplicit_auxiliary_update(
  const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &data,
  std::vector<VectorType *>                                              &dst,
  const std::vector<VectorType *>                                        &src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // Constructor for FEEvaluation objects
  variableContainer<dim, degree, number> variable_list(data,
                                                       *attributes_list,
                                                       global_to_local_solution,
                                                       solveType::NONEXPLICIT_RHS);

  // Initialize, evaluate, and submit based on user function.
  variable_list.eval_local_operator(
    [this](variableContainer<dim, degree, number> &var_list,
           const dealii::Point<dim, size_type>    &q_point_loc)
    {
      this->pde_operator->compute_nonexplicit_RHS(var_list, q_point_loc, current_index);
    },
    dst,
    src,
    cell_range);
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::compute_local_residual(
  const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &data,
  VectorType                                                             &dst,
  [[maybe_unused]] const VectorType                                      &src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // Constructor for FEEvaluation objects
  variableContainer<dim, degree, number> variable_list(data,
                                                       *attributes_list,
                                                       global_to_local_solution,
                                                       solveType::NONEXPLICIT_RHS);

  // Initialize, evaluate, and submit based on user function.
  variable_list.eval_local_operator(
    [this](variableContainer<dim, degree, number> &var_list,
           const dealii::Point<dim, size_type>    &q_point_loc)
    {
      this->pde_operator->compute_nonexplicit_RHS(var_list, q_point_loc, current_index);
    },
    dst,
    src_solution_subset,
    cell_range);
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::compute_local_newton_update(
  const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &data,
  VectorType                                                             &dst,
  const VectorType                                                       &src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // Constructor for FEEvaluation objects
  variableContainer<dim, degree, number> variable_list(data,
                                                       *attributes_list,
                                                       global_to_local_solution,
                                                       solveType::NONEXPLICIT_LHS);

  // Initialize, evaluate, and submit based on user function. Note that the src solution
  // subset must not include the src vector.
  variable_list.eval_local_operator(
    [this](variableContainer<dim, degree, number> &var_list,
           const dealii::Point<dim, size_type>    &q_point_loc)
    {
      this->pde_operator->compute_nonexplicit_LHS(var_list, q_point_loc, current_index);
    },
    dst,
    src,
    src_solution_subset,
    cell_range);
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::compute_diagonal(unsigned int field_index)
{
  inverse_diagonal_entries.reset(new dealii::DiagonalMatrix<VectorType>());
  VectorType &inverse_diagonal = inverse_diagonal_entries->get_vector();
  data->initialize_dof_vector(inverse_diagonal, field_index);
  const unsigned int dummy = 0;
  data->cell_loop(&matrixFreeOperator::local_compute_diagonal,
                  this,
                  inverse_diagonal,
                  dummy);

  set_constrained_entries_to_one(inverse_diagonal);

  for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
    {
      Assert(inverse_diagonal.local_element(i) > 0.0,
             dealii::ExcMessage(
               "No diagonal entry in a positive definite operator should be zero"));
      inverse_diagonal.local_element(i) = 1.0 / inverse_diagonal.local_element(i);
    }
}

template <int dim, int degree, typename number>
void
matrixFreeOperator<dim, degree, number>::local_compute_diagonal(
  const dealii::MatrixFree<dim, number, dealii::VectorizedArray<number>> &data,
  VectorType                                                             &dst,
  [[maybe_unused]] const unsigned int                                    &dummy,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // Constructor for FEEvaluation objects
  variableContainer<dim, degree, number> variable_list(data,
                                                       *attributes_list,
                                                       global_to_local_solution,
                                                       solveType::NONEXPLICIT_LHS);

  // Initialize, evaluate, and submit diagonal based on user function.
  variable_list.eval_local_diagonal(
    [this](variableContainer<dim, degree, number> &var_list,
           const dealii::Point<dim, size_type>    &q_point_loc)
    {
      this->pde_operator->compute_nonexplicit_LHS(var_list, q_point_loc, current_index);
    },
    dst,
    src_solution_subset,
    cell_range);
}

INSTANTIATE_TRI_TEMPLATE(matrixFreeOperator)

PRISMS_PF_END_NAMESPACE
