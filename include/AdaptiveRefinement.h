#ifndef INCLUDE_ADAPTIVEREFINEMENT_H_
#define INCLUDE_ADAPTIVEREFINEMENT_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "fields.h"
#include "userInputParameters.h"

using namespace dealii;

/**
 * A class that handles the determination and application of AMR criterion.
 */
template <int dim, int degree>
class AdaptiveRefinement
{
public:
  using vectorType = dealii::LinearAlgebra::distributed::Vector<double>;

  /**
   * Default constructor.
   */
  AdaptiveRefinement(
    const userInputParameters<dim>            &_userInputs,
    parallel::distributed::Triangulation<dim> &_triangulation,
    std::vector<Field<dim>>                   &_fields,
    std::vector<vectorType *>                 &_solutionSet,
    std::vector<parallel::distributed::SolutionTransfer<dim, vectorType> *> &_soltransSet,
    std::vector<FESystem<dim> *>                                            &_FESet,
    std::vector<DoFHandler<dim> *>                 &_dofHandlersSet_nonconst,
    std::vector<const AffineConstraints<double> *> &_constraintsDirichletSet,
    std::vector<const AffineConstraints<double> *> &_constraintsOtherSet);

  /**
   * Perform the adaptive refinement based on the specified AMR criterion. Also apply
   * constraints when in the 0th timestep.
   */
  void
  do_adaptive_refinement(unsigned int _currentIncrement);

  /**
   * Refine the triangulation and transfer the solution.
   */
  void
  refine_grid();

protected:
  /**
   * Mark cells to be coarsened or refined based on the specified AMR criterion.
   */
  void
  adaptive_refinement_criterion();

private:
  userInputParameters<dim> userInputs;

  parallel::distributed::Triangulation<dim> &triangulation;

  std::vector<Field<dim>> &fields;

  std::vector<vectorType *> &solutionSet;

  std::vector<parallel::distributed::SolutionTransfer<dim, vectorType> *> &soltransSet;

  std::vector<FESystem<dim> *> &FESet;

  std::vector<DoFHandler<dim> *> &dofHandlersSet_nonconst;

  std::vector<const AffineConstraints<double> *> &constraintsDirichletSet;

  std::vector<const AffineConstraints<double> *> &constraintsOtherSet;
};

template <int dim, int degree>
AdaptiveRefinement<dim, degree>::AdaptiveRefinement(
  const userInputParameters<dim>                                          &_userInputs,
  parallel::distributed::Triangulation<dim>                               &_triangulation,
  std::vector<Field<dim>>                                                 &_fields,
  std::vector<vectorType *>                                               &_solutionSet,
  std::vector<parallel::distributed::SolutionTransfer<dim, vectorType> *> &_soltransSet,
  std::vector<FESystem<dim> *>                                            &_FESet,
  std::vector<DoFHandler<dim> *>                 &_dofHandlersSet_nonconst,
  std::vector<const AffineConstraints<double> *> &_constraintsDirichletSet,
  std::vector<const AffineConstraints<double> *> &_constraintsOtherSet)
  : userInputs(_userInputs)
  , triangulation(_triangulation)
  , fields(_fields)
  , solutionSet(_solutionSet)
  , soltransSet(_soltransSet)
  , FESet(_FESet)
  , dofHandlersSet_nonconst(_dofHandlersSet_nonconst)
  , constraintsDirichletSet(_constraintsDirichletSet)
  , constraintsOtherSet(_constraintsOtherSet)
{}

template <int dim, int degree>
void
AdaptiveRefinement<dim, degree>::do_adaptive_refinement(unsigned int currentIncrement)
{
  if (currentIncrement != 0)
    {
      // Apply constraints before remeshing
      for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++)
        {
          constraintsDirichletSet[fieldIndex]->distribute(*solutionSet[fieldIndex]);
          constraintsOtherSet[fieldIndex]->distribute(*solutionSet[fieldIndex]);
          solutionSet[fieldIndex]->update_ghost_values();
        }
    }

  adaptive_refinement_criterion();
  refine_grid();
}

template <int dim, int degree>
void
AdaptiveRefinement<dim, degree>::adaptive_refinement_criterion()
{
  std::vector<std::vector<double>> valuesV;
  std::vector<std::vector<double>> gradientsV;

  QGaussLobatto<dim> quadrature(degree + 1);
  const unsigned int num_quad_points = quadrature.size();

  // Set the update flags
  dealii::UpdateFlags update_flags;
  for (const auto &criterion : userInputs.refinement_criteria)
    {
      if (criterion.criterion_type == VALUE ||
          criterion.criterion_type == VALUE_AND_GRADIENT)
        {
          update_flags = update_values | update_flags;
        }
      else if (criterion.criterion_type == GRADIENT ||
               criterion.criterion_type == VALUE_AND_GRADIENT)
        {
          update_flags = update_gradients | update_flags;
        }
    }

  FEValues<dim> fe_values(*FESet[userInputs.refinement_criteria[0].variable_index],
                          quadrature,
                          update_flags);

  std::vector<double>                         values(num_quad_points);
  std::vector<double>                         gradient_magnitudes(num_quad_points);
  std::vector<dealii::Tensor<1, dim, double>> gradients(num_quad_points);

  typename parallel::distributed::Triangulation<dim>::active_cell_iterator t_cell =
    triangulation.begin_active();

  for (const auto &cell :
       dofHandlersSet_nonconst[userInputs.refinement_criteria[0].variable_index]
         ->active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          for (const auto &criterion : userInputs.refinement_criteria)
            {
              if (update_values & update_flags)
                {
                  fe_values.get_function_values(*solutionSet[criterion.variable_index],
                                                values);
                  valuesV.push_back(values);
                }
              if (update_gradients & update_flags)
                {
                  fe_values.get_function_gradients(*solutionSet[criterion.variable_index],
                                                   gradients);

                  for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
                    {
                      gradient_magnitudes.at(q_point) = gradients.at(q_point).norm();
                    }

                  gradientsV.push_back(gradient_magnitudes);
                }
            }

          bool mark_refine = false;

          for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
            {
              for (const auto &criterion : userInputs.refinement_criteria)
                {
                  if (criterion.criterion_type == VALUE ||
                      criterion.criterion_type == VALUE_AND_GRADIENT)
                    {
                      if ((valuesV[criterion.variable_index][q_point] >
                           criterion.value_lower_bound) &&
                          (valuesV[criterion.variable_index][q_point] <
                           criterion.value_upper_bound))
                        {
                          mark_refine = true;
                          break;
                        }
                    }
                  if (criterion.criterion_type == GRADIENT ||
                      criterion.criterion_type == VALUE_AND_GRADIENT)
                    {
                      if (gradientsV[criterion.variable_index][q_point] >
                          criterion.gradient_lower_bound)
                        {
                          mark_refine = true;
                          break;
                        }
                    }
                }
            }

          valuesV.clear();
          gradientsV.clear();

          // limit the maximal and minimal refinement depth of the mesh
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
  // prepare and refine
  triangulation.prepare_coarsening_and_refinement();
  for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++)
    {
      soltransSet[fieldIndex]->prepare_for_coarsening_and_refinement(
        *solutionSet[fieldIndex]);
    }
  triangulation.execute_coarsening_and_refinement();
}

#endif
