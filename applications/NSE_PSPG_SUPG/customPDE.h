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

  /*void
  makeTriangulation(parallel::distributed::Triangulation<dim> &) const override;*/

  void
  solveIncrement(bool skip_time_dependent) override;

  VectorizedArray<double>
  compute_stabilization_parameter(
    const dealii::Tensor<1, dim, dealii::VectorizedArray<double>> &local_velocity,
    const VectorizedArray<double>                                 &element_volume) const;

  // ================================================================
  // Model constants specific to this subclass
  // ================================================================

  bool fractional_pressure_update_step = false;

  scalarvalueType rho = constV(100.0);
  scalarvalueType mu  = constV(0.01);

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
  const dealii::Tensor<1, dim, dealii::VectorizedArray<double>> &local_velocity,
  const VectorizedArray<double>                                 &element_volume) const
{
  // Norm of the local velocity
  VectorizedArray<double> u_l2norm = 1.0e-12 + local_velocity.norm_square();

  // Stabilization parameter
  VectorizedArray<double> h = std::sqrt(element_volume) * size_modifier;

  VectorizedArray<double> stabilization_parameter =
    constV(1.0) / std::sqrt(time_contribution + constV(4.0) * u_l2norm / h / h);

  return stabilization_parameter;
}

/*
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

template <int dim, int degree>
void
customPDE<dim, degree>::makeTriangulation(
  parallel::distributed::Triangulation<dim> &tria) const
{
  // Mesh parameters
  const double     inner_radius = 1.0;
  const double     outer_radius = 2.0;
  const Point<dim> center(7.0, 2.9);

  const double pad_bottom = 2.9 - outer_radius;
  const double pad_top    = 6.0 - 2.9 - outer_radius;
  const double pad_left   = 7.0 - outer_radius;
  const double pad_right  = 30.0 - 7.0 - outer_radius;

  // Generate the initial mesh with a circular hole
  GridGenerator::plate_with_a_hole(tria,
                                   inner_radius,
                                   outer_radius,
                                   pad_bottom,
                                   pad_top,
                                   pad_left,
                                   pad_right,
                                   center);

  // Mark the boundaries
  for (const auto &cell : tria.active_cell_iterators())
    {
      // Mark the left face with boundary id 0 (inflow), the right with boundary id 1
      // (outflow), and everything else with boundary id 2 (wall). This ensures that the
      // right boundary conditions are used for the benchmark. This reduces the complexity
      // of the code at the cost of flexibility in boundary conditions. For the benchmark
      // case, we don't care about flexibility. If you plan to use this code to create
      // your own triangulation, modify this section accordingly.
      for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
           ++face_number)
        {
          const auto &face = cell->face(face_number);

          if (face->at_boundary())
            {
              if (std::fabs(cell->face(face_number)->center()(0) - 0) < 1e-12)
                {
                  face->set_boundary_id(0);
                }
              else if (std::fabs(cell->face(face_number)->center()(0) -
                                 userInputs.domain_size[0]) < 1e-12)
                {
                  face->set_boundary_id(1);
                }
              else
                {
                  face->set_boundary_id(2);
                }
            }
        }
    }
}*/

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

  // Set fractional_pressure_update_step to false so steps 1 and 2 may occur
  fractional_pressure_update_step = false;

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

  // Set fractional_pressure_update_step to true so steps 3 may occur
  fractional_pressure_update_step = true;

  // Get the RHS of the explicit equations
  if (this->hasExplicitEquation && !skip_time_dependent)
    {
      this->computeExplicitRHS();
    }

  // solve for the projected velocity field
  for (unsigned int fieldIndex = 0; fieldIndex < this->fields.size(); fieldIndex++)
    {
      // Here are the allowed fields that we recalulate
      if (userInputs.var_name[fieldIndex] != "u")
        {
          continue;
        }

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

  if (this->currentIncrement % userInputs.skip_print_steps == 0)
    {
      this->pcout << "wall time: " << time.wall_time() << "s\n";
    }
  // log time
  this->computing_timer.leave_subsection("matrixFreePDE: solveIncrements");
}
