// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <prismspf/core/constraint_handler.h>
#include <prismspf/core/dof_handler.h>
#include <prismspf/core/matrix_free_handler.h>
#include <prismspf/core/matrix_free_operator.h>
#include <prismspf/core/solution_handler.h>
#include <prismspf/core/type_enums.h>
#include <prismspf/core/variable_attributes.h>

#include <prismspf/user_inputs/user_input_parameters.h>

#include <prismspf/solvers/linear_solver_gmg.h>
#include <prismspf/solvers/linear_solver_identity.h>
#include <prismspf/solvers/nonexplicit_base.h>

#include <prismspf/config.h>

#ifdef PRISMS_PF_WITH_CALIPER
#  include <caliper/cali.h>
#endif

PRISMS_PF_BEGIN_NAMESPACE

/**
 * \brief This class handles the self-nonlinear solves of a single nonexplicit field
 */
template <int dim, int degree>
class nonexplicitSelfNonlinearSolver : public nonexplicitBase<dim, degree>
{
public:
  using SystemMatrixType = matrixFreeOperator<dim, degree, double>;

  /**
   * \brief Constructor.
   */
  nonexplicitSelfNonlinearSolver(
    const userInputParameters<dim>                         &_user_inputs,
    const matrixfreeHandler<dim>                           &_matrix_free_handler,
    const triangulationHandler<dim>                        &_triangulation_handler,
    const invmHandler<dim, degree>                         &_invm_handler,
    const constraintHandler<dim>                           &_constraint_handler,
    const dofHandler<dim>                                  &_dof_handler,
    const dealii::MappingQ1<dim>                           &_mapping,
    dealii::MGLevelObject<matrixfreeHandler<dim, float>>   &_mg_matrix_free_handler,
    solutionHandler<dim>                                   &_solution_handler,
    std::shared_ptr<const PDEOperator<dim, degree, double>> _pde_operator,
    std::shared_ptr<const PDEOperator<dim, degree, float>>  _pde_operator_float);

  /**
   * \brief Destructor.
   */
  ~nonexplicitSelfNonlinearSolver() override = default;

  /**
   * \brief Initialize system.
   */
  void
  init() override;

  /**
   * \brief Solve a single update step.
   */
  void
  solve() override;

private:
  /**
   * \brief Map of identity linear solvers
   */
  std::map<unsigned int, std::unique_ptr<identitySolver<dim, degree>>> identity_solvers;

  /**
   * \brief Map of geometric multigrid linear solvers
   */
  std::map<unsigned int, std::unique_ptr<GMGSolver<dim, degree>>> gmg_solvers;

  /**
   * \brief PDE operator but for floats!
   */
  std::shared_ptr<const PDEOperator<dim, degree, float>> pde_operator_float;
};

PRISMS_PF_END_NAMESPACE