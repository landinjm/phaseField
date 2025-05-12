// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <prismspf/solvers/linear_solver_gmg.h>
#include <prismspf/solvers/linear_solver_identity.h>
#include <prismspf/solvers/nonexplicit_base.h>

#include <prismspf/config.h>

#ifdef PRISMS_PF_WITH_CALIPER
#  include <caliper/cali.h>
#endif

PRISMS_PF_BEGIN_NAMESPACE

/**
 * \brief This class handles the co-nonlinear solves of a several nonexplicit fields
 */
template <unsigned int dim, unsigned int degree>
class nonexplicitCoNonlinearSolver : public nonexplicitBase<dim, degree>
{
public:
  using SystemMatrixType = matrixFreeOperator<dim, degree, double>;
  using VectorType       = dealii::LinearAlgebra::distributed::Vector<double>;

  /**
   * \brief Constructor.
   */
  nonexplicitCoNonlinearSolver(
    const userInputParameters<dim>                         &_user_inputs,
    const matrixfreeHandler<dim, double>                   &_matrix_free_handler,
    const triangulationHandler<dim>                        &_triangulation_handler,
    const invmHandler<dim, degree, double>                 &_invm_handler,
    const constraintHandler<dim, degree>                   &_constraint_handler,
    const dofHandler<dim>                                  &_dof_handler,
    const dealii::MappingQ1<dim>                           &_mapping,
    dealii::MGLevelObject<matrixfreeHandler<dim, float>>   &_mg_matrix_free_handler,
    solutionHandler<dim>                                   &_solution_handler,
    std::shared_ptr<const PDEOperator<dim, degree, double>> _pde_operator,
    std::shared_ptr<const PDEOperator<dim, degree, float>>  _pde_operator_float,
    const MGInfo<dim>                                      &_mg_info);

  /**
   * \brief Destructor.
   */
  ~nonexplicitCoNonlinearSolver() override = default;

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
   * \brief Mapping from global solution vectors to the local ones
   */
  std::map<unsigned int, std::map<std::pair<unsigned int, dependencyType>, unsigned int>>
    global_to_local_solution;

  /**
   * \brief Subset of solutions fields that are necessary for explicit solves.
   */
  std::map<unsigned int, std::vector<VectorType *>> solution_subset;

  /**
   * \brief Subset of new solutions fields that are necessary for explicit solves.
   */
  std::map<unsigned int, std::vector<VectorType *>> new_solution_subset;

  /**
   * \brief List of subset attributes.
   */
  std::vector<std::map<unsigned int, variableAttributes>> subset_attributes_list;

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

  /**
   * \brief Multigrid information
   */
  const MGInfo<dim> *mg_info;
};

PRISMS_PF_END_NAMESPACE