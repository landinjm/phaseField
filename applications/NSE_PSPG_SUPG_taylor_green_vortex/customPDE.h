#include "matrixFreePDE.h"

using namespace dealii;

template <int dim, int degree>
class customPDE : public MatrixFreePDE<dim, degree>
{
public:
  // Constructor
  customPDE(userInputParameters<dim> _userInputs)
    : MatrixFreePDE<dim, degree>(_userInputs)
    , userInputs(_userInputs) {};

  // Function to set the initial conditions (in ICs_and_BCs.h)
  void
  setInitialCondition([[maybe_unused]] const Point<dim>  &p,
                      [[maybe_unused]] const unsigned int index,
                      [[maybe_unused]] double            &scalar_IC,
                      [[maybe_unused]] Vector<double>    &vector_IC) override;

  // Function to set the non-uniform Dirichlet boundary conditions (in
  // ICs_and_BCs.h)
  void
  setNonUniformDirichletBCs([[maybe_unused]] const Point<dim>  &p,
                            [[maybe_unused]] const unsigned int index,
                            [[maybe_unused]] const unsigned int direction,
                            [[maybe_unused]] const double       time,
                            [[maybe_unused]] double            &scalar_BC,
                            [[maybe_unused]] Vector<double>    &vector_BC) override;

private:
#include "typeDefs.h"

  const userInputParameters<dim> userInputs;

  // Function to set the RHS of the governing equations for explicit time
  // dependent equations (in equations.h)
  void
  explicitEquationRHS(
    [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>>
                                                              &variable_list,
    [[maybe_unused]] const Point<dim, VectorizedArray<double>> q_point_loc,
    [[maybe_unused]] const VectorizedArray<double> element_volume) const override;

  // Function to set the RHS of the governing equations for all other equations
  // (in equations.h)
  void
  nonExplicitEquationRHS(
    [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>>
                                                              &variable_list,
    [[maybe_unused]] const Point<dim, VectorizedArray<double>> q_point_loc,
    [[maybe_unused]] const VectorizedArray<double> element_volume) const override;

  // Function to set the LHS of the governing equations (in equations.h)
  void
  equationLHS(
    [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>>
                                                              &variable_list,
    [[maybe_unused]] const Point<dim, VectorizedArray<double>> q_point_loc,
    [[maybe_unused]] const VectorizedArray<double> element_volume) const override;

// Function to set postprocessing expressions (in postprocess.h)
#ifdef POSTPROCESS_FILE_EXISTS
  void
  postProcessedFields(
    [[maybe_unused]] const variableContainer<dim, degree, VectorizedArray<double>>
      &variable_list,
    [[maybe_unused]] variableContainer<dim, degree, VectorizedArray<double>>
                                                              &pp_variable_list,
    [[maybe_unused]] const Point<dim, VectorizedArray<double>> q_point_loc,
    [[maybe_unused]] const VectorizedArray<double> element_volume) const override;
#endif

// Function to set the nucleation probability (in nucleation.h)
#ifdef NUCLEATION_FILE_EXISTS
  double
  getNucleationProbability([[maybe_unused]] variableValueContainer variable_value,
                           [[maybe_unused]] double                 dV) const override;
#endif

  // ================================================================
  // Methods specific to this subclass
  // ================================================================

  void
  solveIncrement(bool skip_time_dependent) override;

  VectorizedArray<double>
  compute_stabilization_parameter(
    const Tensor<1, dim, VectorizedArray<double>> &local_velocity,
    const VectorizedArray<double>                 &element_volume) const;

  // ================================================================
  // Model constants specific to this subclass
  // ================================================================

  scalarvalueType rho = constV(1.0);
  scalarvalueType mu  = constV(0.1);

  scalarvalueType nu = mu / rho;
  scalarvalueType dt = constV(userInputs.dtValue);

  // Values for stabilization term
  scalarvalueType size_modifier = constV(std::sqrt(4.0 / M_PI) / degree);
  scalarvalueType time_contribution =
    constV(dealii::Utilities::fixed_power<2>(1.0 / userInputs.dtValue));

  // ================================================================
};

template <int dim, int degree>
VectorizedArray<double>
customPDE<dim, degree>::compute_stabilization_parameter(
  const Tensor<1, dim, VectorizedArray<double>> &local_velocity,
  const VectorizedArray<double>                 &element_volume) const
{
  // Norm of the local velocity
  VectorizedArray<double> u_l2norm = 1.0e-12 + local_velocity.norm_square();

  // Stabilization parameter
  VectorizedArray<double> h = std::sqrt(element_volume) * size_modifier;

  VectorizedArray<double> stabilization_parameter =
    constV(1.0) /
    std::sqrt(time_contribution + constV(4.0) * u_l2norm / h / h +
              constV(9.0) * Utilities::fixed_power<2>(constV(4.0) * nu / h / h));

  return stabilization_parameter;
}

// =================================================================================
// Function overriding solveIncrement
// =================================================================================
#include <deal.II/lac/solver_cg.h>

template <int dim, int degree>
void
customPDE<dim, degree>::solveIncrement(bool skip_time_dependent)
{
  // log time
  this->computing_timer.enter_subsection("matrixFreePDE: solveIncrements");
  Timer time;
  char  buffer[200];

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
      if (this->fields[fieldIndex].pdetype == EXPLICIT_TIME_DEPENDENT &&
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

              if (!numbers::is_finite(solution_L2_norm))
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
      bool         nonlinear_it_converged = false;
      unsigned int nonlinear_it_index     = 0;

      while (!nonlinear_it_converged)
        {
          nonlinear_it_converged = true; // Set to true here and will be set to false if
                                         // any variable isn't converged

          for (unsigned int fieldIndex = 0; fieldIndex < this->fields.size();
               fieldIndex++)
            {
              this->currentFieldIndex = fieldIndex; // Used in computeLHS()

              // Compute RHS if we have a nonexplicit field
              if ((this->fields[fieldIndex].pdetype == IMPLICIT_TIME_DEPENDENT &&
                   !skip_time_dependent) ||
                  this->fields[fieldIndex].pdetype == TIME_INDEPENDENT ||
                  this->fields[fieldIndex].pdetype == AUXILIARY)
                {
                  this->computeNonexplicitRHS();
                }

              if ((this->fields[fieldIndex].pdetype == IMPLICIT_TIME_DEPENDENT &&
                   !skip_time_dependent) ||
                  this->fields[fieldIndex].pdetype == TIME_INDEPENDENT)
                {
                  if (this->currentIncrement % userInputs.skip_print_steps == 0 &&
                      userInputs.var_nonlinear[fieldIndex])
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

                  nonlinear_it_converged =
                    this->updateImplicitSolution(fieldIndex, nonlinear_it_index);

                  // Apply Boundary conditions
                  this->applyBCs(fieldIndex);
                }
              else if (this->fields[fieldIndex].pdetype == AUXILIARY)
                {
                  if (userInputs.var_nonlinear[fieldIndex] || nonlinear_it_index == 0)
                    {
                      // If the equation for this field is nonlinear, save the
                      // old solution
                      if (userInputs.var_nonlinear[fieldIndex])
                        {
                          if (this->fields[fieldIndex].type == SCALAR)
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
                      if (userInputs.var_nonlinear[fieldIndex])
                        {
                          if (MatrixFreePDE<dim, degree>::userInputs
                                .nonlinear_solver_parameters.getToleranceType(
                                  fieldIndex) == ABSOLUTE_SOLUTION_CHANGE)
                            {
                              double diff;

                              if (this->fields[fieldIndex].type == SCALAR)
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
                                              << diff << " " << nonlinear_it_index << " "
                                              << this->currentIncrement << std::endl;
                                }

                              if (diff > MatrixFreePDE<dim, degree>::userInputs
                                           .nonlinear_solver_parameters.getToleranceValue(
                                             fieldIndex) &&
                                  nonlinear_it_index <
                                    MatrixFreePDE<dim, degree>::userInputs
                                      .nonlinear_solver_parameters.getMaxIterations())
                                {
                                  nonlinear_it_converged = false;
                                }
                            }
                          else
                            {
                              std::cerr << "PRISMS-PF Error: Nonlinear solver "
                                           "tolerance types other than ABSOLUTE_CHANGE "
                                           "have yet to be implemented."
                                        << std::endl;
                            }
                        }
                    }
                }

              // check if solution is nan
              if (!numbers::is_finite(this->solutionSet[fieldIndex]->l2_norm()))
                {
                  snprintf(buffer,
                           sizeof(buffer),
                           "ERROR: field '%s' solution is NAN. exiting.\n\n",
                           this->fields[fieldIndex].name.c_str());
                  this->pcout << buffer;
                  exit(-1);
                }
            }

          nonlinear_it_index++;
        }
    }

  if (this->currentIncrement % userInputs.skip_print_steps == 0)
    {
      this->pcout << "wall time: " << time.wall_time() << "s\n";
    }
  // log time
  this->computing_timer.leave_subsection("matrixFreePDE: solveIncrements");
}
