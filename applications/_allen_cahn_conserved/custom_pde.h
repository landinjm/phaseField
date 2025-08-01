// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <core/matrixFreePDE.h>

using namespace dealii;

template <int dim, int degree>
class CustomPDE : public MatrixFreePDE<dim, degree>
{
public:
  // Constructor
  CustomPDE(UserInputParameters<dim> _userInputs)
    : MatrixFreePDE<dim, degree>(_userInputs)
    , userInputs(_userInputs) {};

  // Function to set the initial conditions (in ICs_and_BCs.h)
  void
  setInitialCondition([[maybe_unused]] const Point<dim>  &p,
                      [[maybe_unused]] const unsigned int index,
                      [[maybe_unused]] number            &scalar_IC,
                      [[maybe_unused]] Vector<double>    &vector_IC) override;

  // Function to set the non-uniform Dirichlet boundary conditions (in
  // ICs_and_BCs.h)
  void
  setNonUniformDirichletBCs([[maybe_unused]] const Point<dim>  &p,
                            [[maybe_unused]] const unsigned int index,
                            [[maybe_unused]] const unsigned int direction,
                            [[maybe_unused]] const number       time,
                            [[maybe_unused]] number            &scalar_BC,
                            [[maybe_unused]] Vector<double>    &vector_BC) override;

private:
#include <core/typeDefs.h>

  const UserInputParameters<dim> userInputs;

  // Function to set the RHS of the governing equations for explicit time
  // dependent equations (in equations.cc)
  void
  explicitEquationRHS(
    [[maybe_unused]] VariableContainer<dim, degree, VectorizedArray<double>>
                                                              &variable_list,
    [[maybe_unused]] const Point<dim, VectorizedArray<double>> q_point_loc,
    [[maybe_unused]] const VectorizedArray<double> element_volume) const override;

  // Function to set the RHS of the governing equations for all other equations
  // (in equations.h)
  void
  nonExplicitEquationRHS(
    [[maybe_unused]] VariableContainer<dim, degree, VectorizedArray<double>>
                                                              &variable_list,
    [[maybe_unused]] const Point<dim, VectorizedArray<double>> q_point_loc,
    [[maybe_unused]] const VectorizedArray<double> element_volume) const override;

  // Function to set the LHS of the governing equations (in equations.cc)
  void
  equationLHS(
    [[maybe_unused]] VariableContainer<dim, degree, VectorizedArray<double>>
                                                              &variable_list,
    [[maybe_unused]] const Point<dim, VectorizedArray<double>> q_point_loc,
    [[maybe_unused]] const VectorizedArray<double> element_volume) const override;

// Function to set postprocessing expressions (in postprocess.cc)
#ifdef POSTPROCESS_FILE_EXISTS
  void
  postProcessedFields(
    [[maybe_unused]] const VariableContainer<dim, degree, VectorizedArray<double>>
      &variable_list,
    [[maybe_unused]] VariableContainer<dim, degree, VectorizedArray<double>>
                                                              &pp_variable_list,
    [[maybe_unused]] const Point<dim, VectorizedArray<double>> q_point_loc,
    [[maybe_unused]] const VectorizedArray<double> element_volume) const override;
#endif

// Function to set the nucleation probability (in nucleation.cc)
#ifdef NUCLEATION_FILE_EXISTS
  double
  getNucleationProbability([[maybe_unused]] variableValueContainer variable_value,
                           [[maybe_unused]] number                 dV) const override;
#endif

  // ================================================================
  // Methods specific to this subclass
  // ================================================================

  // Function to override solveIncrement from
  // ../../src/matrixfree/solveIncrement.cc
  void
  solveIncrement(bool skip_time_dependent) override;

  // ================================================================
  // Model constants specific to this subclass
  // ================================================================

  double MnV = userInputs.get_model_constant_double("MnV");
  double KnV = userInputs.get_model_constant_double("KnV");

  double integrated_mu;
  double integrated_n;

  // ================================================================
};

// =================================================================================
// Function overriding solveIncrement
// =================================================================================
// solve each time increment
#include <deal.II/lac/solver_cg.h>

#include <core/exceptions.h>

template <int dim, int degree>
void
CustomPDE<dim, degree>::solveIncrement(bool skip_time_dependent)
{
  // log time
  this->computing_timer.enter_subsection("matrixFreePDE: solveIncrements");
  Timer time;
  char  buffer[200];

  // Calculating integral for mu (field 1)
  this->Integrator(integrated_n, 0, this->solutionSet);
  this->Integrator(integrated_mu, 1, this->solutionSet);

  if (this->currentIncrement % userInputs.skip_print_steps == 0)
    {
      snprintf(buffer, sizeof(buffer), "Integrated mu is %12.6e\n", integrated_mu);
      this->pcout << buffer;
    }

  // Get the RHS of the explicit equations
  if (this->hasExplicitEquation && !skip_time_dependent)
    {
      this->computeExplicitRHS();
    }

  // solve for each field
  for (unsigned int fieldIndex = 0; fieldIndex < this->fields.size(); fieldIndex++)
    {
      this->currentFieldIndex = fieldIndex; // Used in computeLHS()

      // Parabolic (first order derivatives in time) fields
      if (this->fields[fieldIndex].pdetype == ExplicitTimeDependent &&
          !skip_time_dependent)
        {
          this->updateExplicitSolution(fieldIndex);

          // Apply Boundary conditions
          this->applyBCs(fieldIndex);

          // Print update to screen and confirm that solution isn't nan
          if (this->currentIncrement % userInputs.skip_print_steps == 0)
            {
              double solution_L2_norm = this->solutionSet[fieldIndex]->l2_norm();

              snprintf(buffer,
                       sizeof(buffer),
                       "field '%2s' [explicit solve]: current solution: "
                       "%12.6e, current residual:%12.6e\n",
                       this->fields[fieldIndex].name.c_str(),
                       solution_L2_norm,
                       this->residualSet[fieldIndex]->l2_norm());
              this->pcout << buffer;

              if (!Numbers::is_finite(solution_L2_norm))
                {
                  snprintf(buffer,
                           sizeof(buffer),
                           "ERROR: field '%s' solution is NAN. exiting.\n\n",
                           this->fields[fieldIndex].name.c_str());
                  this->pcout << buffer;
                  exit(-1);
                }
            }
        }
    }

  // Now, update the non-explicit variables
  // For the time being, this is just the elliptic equations, but implicit
  // parabolic and auxilary equations should also be here
  if (this->hasNonExplicitEquation)
    {
      bool         nonlinear_iteration_converged = false;
      unsigned int nonlinear_iteration_index     = 0;

      while (!nonlinear_iteration_converged)
        {
          nonlinear_iteration_converged = true; // Set to true here and will be set to
                                                // false if any variable isn't converged

          // Update residualSet for the non-explicitly updated variables
          // compute_nonexplicit_rhs()
          // Ideally, I'd just do this for the non-explicit variables, but for
          // now I'll do all of them this is a little redundant, but hopefully
          // not too terrible
          this->computeNonexplicitRHS();

          for (unsigned int fieldIndex = 0; fieldIndex < this->fields.size();
               fieldIndex++)
            {
              this->currentFieldIndex = fieldIndex; // Used in computeLHS()

              if ((this->fields[fieldIndex].pdetype == ImplicitTimeDependent &&
                   !skip_time_dependent) ||
                  this->fields[fieldIndex].pdetype == TimeIndependent)
                {
                  if (this->currentIncrement % userInputs.skip_print_steps == 0 &&
                      this->var_attributes.at(fieldIndex).is_nonlinear)
                    {
                      snprintf(buffer,
                               sizeof(buffer),
                               "field '%2s' [nonlinear solve]: current "
                               "solution: %12.6e, current residual:%12.6e\n",
                               this->fields[fieldIndex].name.c_str(),
                               this->solutionSet[fieldIndex]->l2_norm(),
                               this->residualSet[fieldIndex]->l2_norm());
                      this->pcout << buffer;
                    }

                  nonlinear_iteration_converged =
                    this->updateImplicitSolution(fieldIndex, nonlinear_iteration_index);

                  // Apply Boundary conditions
                  this->applyBCs(fieldIndex);
                }
              else if (this->fields[fieldIndex].pdetype == Auxiliary)
                {
                  if (this->var_attributes.at(fieldIndex).is_nonlinear ||
                      nonlinear_iteration_index == 0)
                    {
                      // If the equation for this field is nonlinear, save the
                      // old solution
                      if (this->var_attributes.at(fieldIndex).is_nonlinear)
                        {
                          if (this->fields[fieldIndex].type == Scalar)
                            {
                              this->dU_scalar = *this->solutionSet[fieldIndex];
                            }
                          else
                            {
                              this->dU_vector = *this->solutionSet[fieldIndex];
                            }
                        }

                      this->updateExplicitSolution(fieldIndex);

                      // Apply Boundary conditions
                      this->applyBCs(fieldIndex);

                      // Print update to screen
                      if (this->currentIncrement % userInputs.skip_print_steps == 0)
                        {
                          snprintf(buffer,
                                   sizeof(buffer),
                                   "field '%2s' [auxiliary solve]: current solution: "
                                   "%12.6e, current residual:%12.6e\n",
                                   this->fields[fieldIndex].name.c_str(),
                                   this->solutionSet[fieldIndex]->l2_norm(),
                                   this->residualSet[fieldIndex]->l2_norm());
                          this->pcout << buffer;
                        }

                      // Check to see if this individual variable has converged
                      if (this->var_attributes.at(fieldIndex).is_nonlinear)
                        {
                          if (MatrixFreePDE<dim, degree>::userInputs
                                .nonlinear_solver_parameters.getToleranceType(
                                  fieldIndex) == ABSOLUTE_SOLUTION_CHANGE)
                            {
                              double diff;

                              if (this->fields[fieldIndex].type == Scalar)
                                {
                                  this->dU_scalar -= *this->solutionSet[fieldIndex];
                                  diff = this->dU_scalar.l2_norm();
                                }
                              else
                                {
                                  this->dU_vector -= *this->solutionSet[fieldIndex];
                                  diff = this->dU_vector.l2_norm();
                                }
                              if (this->currentIncrement % userInputs.skip_print_steps ==
                                  0)
                                {
                                  this->pcout << "Relative difference between nonlinear "
                                                 "iterations: "
                                              << diff << " " << nonlinear_iteration_index
                                              << " " << this->currentIncrement
                                              << std::endl;
                                }

                              if (diff > MatrixFreePDE<dim, degree>::userInputs
                                           .nonlinear_solver_parameters.getToleranceValue(
                                             fieldIndex) &&
                                  nonlinear_iteration_index <
                                    MatrixFreePDE<dim, degree>::userInputs
                                      .nonlinear_solver_parameters.getMaxIterations())
                                {
                                  nonlinear_iteration_converged = false;
                                }
                            }
                          else
                            {
                              AssertThrow(
                                false,
                                FeatureNotImplemented(
                                  "Nonlinear solver tolerances besides ABSOLUTE_CHANGE"));
                            }
                        }
                    }
                }

              // check if solution is nan
              if (!Numbers::is_finite(this->solutionSet[fieldIndex]->l2_norm()))
                {
                  snprintf(buffer,
                           sizeof(buffer),
                           "ERROR: field '%s' solution is NAN. exiting.\n\n",
                           this->fields[fieldIndex].name.c_str());
                  this->pcout << buffer;
                  exit(-1);
                }
            }

          nonlinear_iteration_index++;
        }
    }

  if (this->currentIncrement % userInputs.skip_print_steps == 0)
    {
      this->pcout << "wall time: " << time.wall_time() << "s\n";
    }
  // log time
  this->computing_timer.leave_subsection("matrixFreePDE: solveIncrements");
}