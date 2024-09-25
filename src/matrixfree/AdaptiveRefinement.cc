#include "../../include/AdaptiveRefinement.h"

using namespace dealii;

template <int dim, int degree>
AdaptiveRefinement<dim, degree>::AdaptiveRefinement(
  const userInputParameters<dim>            &_userInputs,
  parallel::distributed::Triangulation<dim> &_triangulation,
  std::vector<Field<dim>>                   &_fields,
  std::vector<vectorType *>                 &_solution_set,
  std::vector<parallel::distributed::SolutionTransfer<dim, vectorType> *>
                                                 &_solution_transfer_set,
  std::vector<std::unique_ptr<FESystem<dim>>>    &_FE_set,
  std::vector<DoFHandler<dim> *>                 &_dof_handler_set_nonconst,
  std::vector<const AffineConstraints<double> *> &_constraintsDirichletSet,
  std::vector<const AffineConstraints<double> *> &_constraintsOtherSet)
  : userInputs(_userInputs)
  , triangulation(_triangulation)
  , fields(_fields)
  , solution_set(_solution_set)
  , solution_transfer_set(_solution_transfer_set)
  , FE_set(_FE_set)
  , dof_handler_set_nonconst(_dof_handler_set_nonconst)
  , constraintsDirichletSet(_constraintsDirichletSet)
  , constraintsOtherSet(_constraintsOtherSet)
{}

template <int dim, int degree>
void
AdaptiveRefinement<dim, degree>::do_adaptive_refinement(unsigned int currentIncrement)
{
  // Apply constraints for the initial condition so they are considered when remeshing
  if (currentIncrement != 0)
    {
      for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++)
        {
          constraintsDirichletSet[fieldIndex]->distribute(*solution_set[fieldIndex]);
          constraintsOtherSet[fieldIndex]->distribute(*solution_set[fieldIndex]);
          solution_set[fieldIndex]->update_ghost_values();
        }
    }

  adaptive_refinement_criterion();
  refine_grid();
}

template <int dim, int degree>
void
AdaptiveRefinement<dim, degree>::adaptive_refinement_criterion()
{
  QGaussLobatto<dim> quadrature(degree + 1);
  const unsigned int num_quad_points = quadrature.size();

  // Set the update flags
  dealii::UpdateFlags update_flags;
  for (const auto &criterion : userInputs.refinement_criteria)
    {
      if (criterion.criterion_type & criterion_value)
        {
          update_flags |= update_values;
        }
      else if (criterion.criterion_type & criterion_gradient)
        {
          update_flags |= update_gradients;
        }
    }

  FEValues<dim> fe_values(*FE_set[userInputs.refinement_criteria[0].variable_index],
                          quadrature,
                          update_flags);

  std::vector<double>                         values(num_quad_points);
  std::vector<double>                         gradient_magnitudes(num_quad_points);
  std::vector<dealii::Tensor<1, dim, double>> gradients(num_quad_points);

  typename parallel::distributed::Triangulation<dim>::active_cell_iterator t_cell =
    triangulation.begin_active();

  for (const auto &cell :
       dof_handler_set_nonconst[userInputs.refinement_criteria[0].variable_index]
         ->active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          bool mark_refine = false;

          // Loop through the refinement criteria to determine whether a cell needs to be
          // refined or coarsened
          for (const auto &criterion : userInputs.refinement_criteria)
            {
              // Get the values and/or gradients
              if (update_values & update_flags)
                {
                  fe_values.get_function_values(*solution_set[criterion.variable_index],
                                                values);
                }
              if (update_gradients & update_flags)
                {
                  fe_values.get_function_gradients(
                    *solution_set[criterion.variable_index],
                    gradients);

                  for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
                    {
                      gradient_magnitudes[q_point] = gradients[q_point].norm();
                    }
                }

              // Loop through the quadrature points and determine if the cell needs to be
              // refined
              for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
                {
                  if (criterion.criterion_type & criterion_value &&
                      values[q_point] > criterion.value_lower_bound &&
                      values[q_point] < criterion.value_upper_bound)
                    {
                      mark_refine = true;
                      break;
                    }
                  if (criterion.criterion_type & criterion_gradient &&
                      gradient_magnitudes[q_point] > criterion.gradient_lower_bound)
                    {
                      mark_refine = true;
                      break;
                    }
                }

              // Early exit for when there are multiple refinement criteria
              if (mark_refine)
                {
                  break;
                }
            }

          // Limit the max and min refinement depth of the mesh
          unsigned int current_level = t_cell->level();

          if ((mark_refine && current_level < userInputs.max_refinement_level))
            {
              cell->set_refine_flag();
            }
          else if (!mark_refine && current_level > userInputs.min_refinement_level)
            {
              cell->set_coarsen_flag();
            }
        }
      ++t_cell;
    }
}

template <int dim, int degree>
void
AdaptiveRefinement<dim, degree>::refine_grid()
{
  // Prepare for refinement
  triangulation.prepare_coarsening_and_refinement();

  // Transfer solution
  for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++)
    {
      solution_transfer_set[fieldIndex]->prepare_for_coarsening_and_refinement(
        *solution_set[fieldIndex]);
    }

  // Execute refinement
  triangulation.execute_coarsening_and_refinement();
}

// Explicit instantiation
template class AdaptiveRefinement<2, 1>;
template class AdaptiveRefinement<2, 2>;
template class AdaptiveRefinement<2, 3>;
template class AdaptiveRefinement<2, 4>;
template class AdaptiveRefinement<3, 1>;
template class AdaptiveRefinement<3, 2>;
template class AdaptiveRefinement<3, 3>;
template class AdaptiveRefinement<3, 4>;