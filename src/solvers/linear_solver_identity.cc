// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <deal.II/lac/solver_cg.h>

#include <prismspf/core/conditional_ostreams.h>
#include <prismspf/core/constraint_handler.h>
#include <prismspf/core/matrix_free_handler.h>
#include <prismspf/core/matrix_free_operator.h>
#include <prismspf/core/pde_operator.h>
#include <prismspf/core/solution_handler.h>
#include <prismspf/core/type_enums.h>
#include <prismspf/core/variable_attributes.h>

#include <prismspf/user_inputs/user_input_parameters.h>

#include <prismspf/solvers/linear_solver_base.h>
#include <prismspf/solvers/linear_solver_identity.h>

#include <prismspf/config.h>

#include <memory>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
IdentitySolver<dim, degree, number>::IdentitySolver(
  const UserInputParameters<dim>                         &_user_inputs,
  const VariableAttributes                               &_variable_attributes,
  const MatrixfreeHandler<dim, number>                   &_matrix_free_handler,
  const ConstraintHandler<dim, degree, number>           &_constraint_handler,
  SolutionHandler<dim, number>                           &_solution_handler,
  std::shared_ptr<const PDEOperator<dim, degree, number>> _pde_operator)
  : LinearSolverBase<dim, degree, number>(_user_inputs,
                                          _variable_attributes,
                                          _matrix_free_handler,
                                          _constraint_handler,
                                          _solution_handler,
                                          std::move(_pde_operator))
{}

template <unsigned int dim, unsigned int degree, typename number>
void
IdentitySolver<dim, degree, number>::init()
{
  this->get_system_matrix()->clear();
  this->get_system_matrix()->initialize(
    this->get_matrix_free_handler().get_matrix_free());
  this->get_update_system_matrix()->clear();
  this->get_update_system_matrix()->initialize(
    this->get_matrix_free_handler().get_matrix_free());

  this->get_system_matrix()->add_global_to_local_mapping(
    this->get_residual_global_to_local_solution());
  this->get_system_matrix()->add_src_solution_subset(this->get_residual_src());

  this->get_update_system_matrix()->add_global_to_local_mapping(
    this->get_newton_update_global_to_local_solution());
  this->get_update_system_matrix()->add_src_solution_subset(
    this->get_newton_update_src());

  // Apply constraints
  this->get_constraint_handler()
    .get_constraint(this->get_field_index())
    .distribute(
      *(this->get_solution_handler().get_solution_vector(this->get_field_index(),
                                                         DependencyType::Normal)));
}

template <unsigned int dim, unsigned int degree, typename number>
void
IdentitySolver<dim, degree, number>::reinit()
{
  // Apply constraints
  this->get_constraint_handler()
    .get_constraint(this->get_field_index())
    .distribute(
      *(this->get_solution_handler().get_solution_vector(this->get_field_index(),
                                                         DependencyType::Normal)));
}

template <unsigned int dim, unsigned int degree, typename number>
void
IdentitySolver<dim, degree, number>::solve(const number &step_length)
{
  auto *solution =
    this->get_solution_handler().get_solution_vector(this->get_field_index(),
                                                     DependencyType::Normal);

  // Compute the residual
  this->get_system_matrix()->compute_residual(*this->get_residual(), *solution);
  if (this->get_user_inputs().get_output_parameters().should_output(
        this->get_user_inputs().get_temporal_discretization().get_increment()))
    {
      ConditionalOStreams::pout_summary()
        << "  field: " << this->get_field_index()
        << " Initial residual: " << this->get_residual()->l2_norm() << std::flush;
    }

  // Determine the residual tolerance
  this->compute_solver_tolerance();

  // Update solver controls
  this->get_solver_control().set_tolerance(this->get_tolerance());
  dealii::SolverCG<VectorType> cg_solver(this->get_solver_control());

  try
    {
      *this->get_newton_update() = 0.0;
      cg_solver.solve(*(this->get_update_system_matrix()),
                      *(this->get_newton_update()),
                      *(this->get_residual()),
                      dealii::PreconditionIdentity());
    }
  catch (...)
    {
      ConditionalOStreams::pout_base()
        << "Warning: linear solver did not converge as per set tolerances.\n";
    }
  this->get_constraint_handler()
    .get_constraint(this->get_field_index())
    .set_zero(*this->get_newton_update());

  if (this->get_user_inputs().get_output_parameters().should_output(
        this->get_user_inputs().get_temporal_discretization().get_increment()))
    {
      ConditionalOStreams::pout_summary()
        << " Final residual: " << this->get_solver_control().last_value()
        << " Steps: " << this->get_solver_control().last_step() << "\n"
        << std::flush;
    }

  // Update the solutions
  (*solution).add(step_length, *this->get_newton_update());

  // Apply constraints
  // This may be redundant with the constraints on the update step.
  this->get_constraint_handler()
    .get_constraint(this->get_field_index())
    .distribute(*solution);
}

#include "solvers/linear_solver_identity.inst"

PRISMS_PF_END_NAMESPACE
