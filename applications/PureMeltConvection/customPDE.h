#include "../../include/matrixFreePDE.h"

template <int dim, int degree>
class customPDE : public MatrixFreePDE<dim, degree> {
public:
    // Constructor
    customPDE(userInputParameters<dim> _userInputs)
        : MatrixFreePDE<dim, degree>(_userInputs)
        , userInputs(_userInputs) { };

    // Function to set the initial conditions (in ICs_and_BCs.h)
    void setInitialCondition(const dealii::Point<dim>& p, const unsigned int index, double& scalar_IC, dealii::Vector<double>& vector_IC);

    // Function to set the non-uniform Dirichlet boundary conditions (in ICs_and_BCs.h)
    void setNonUniformDirichletBCs(const dealii::Point<dim>& p, const unsigned int index, const unsigned int direction, const double time, double& scalar_BC, dealii::Vector<double>& vector_BC);

private:
#include "../../include/typeDefs.h"

    const userInputParameters<dim> userInputs;

    // Function to set the RHS of the governing equations for explicit time dependent equations (in equations.h)
    void explicitEquationRHS(variableContainer<dim, degree, dealii::VectorizedArray<double>>& variable_list,
        dealii::Point<dim, dealii::VectorizedArray<double>> q_point_loc) const;

    // Function to set the RHS of the governing equations for all other equations (in equations.h)
    void nonExplicitEquationRHS(variableContainer<dim, degree, dealii::VectorizedArray<double>>& variable_list,
        dealii::Point<dim, dealii::VectorizedArray<double>> q_point_loc) const;

    // Function to set the LHS of the governing equations (in equations.h)
    void equationLHS(variableContainer<dim, degree, dealii::VectorizedArray<double>>& variable_list,
        dealii::Point<dim, dealii::VectorizedArray<double>> q_point_loc) const;

// Function to set postprocessing expressions (in postprocess.h)
#ifdef POSTPROCESS_FILE_EXISTS
    void postProcessedFields(const variableContainer<dim, degree, dealii::VectorizedArray<double>>& variable_list,
        variableContainer<dim, degree, dealii::VectorizedArray<double>>& pp_variable_list,
        const dealii::Point<dim, dealii::VectorizedArray<double>> q_point_loc) const;
#endif

// Function to set the nucleation probability (in nucleation.h)
#ifdef NUCLEATION_FILE_EXISTS
    double getNucleationProbability(variableValueContainer variable_value, double dV) const;
#endif

    // ================================================================
    // Methods specific to this subclass
    // ================================================================

    // Function to override solveIncrement from ../../src/matrixfree/solveIncrement.cc
    void solveIncrement(bool skip_time_dependent);

    // ================================================================
    // Model constants specific to this subclass
    // ================================================================

    double nu = userInputs.get_model_constant_double("nu");
    double rho = userInputs.get_model_constant_double("rho");
    int switchFractional = userInputs.get_model_constant_int("switchToFractional");

    // Change var type
    unsigned int switchToFractional = switchFractional;

    // This bool acts as a switch to indicate what Chorin projection step is being calculating
    bool ChorinSwitch;

    double h = 2.757;
    double D = 4;
    double reg = 1e-10;
    double eps4 = 0.05;
    double tau = 1.0;
    double W = 1.0;
    double lambda = 1.5957 * D * tau / W / W;
    double delta = W / std::sqrt(2.0);

    // ================================================================
};

// =================================================================================
// Function overriding solveIncrement
// =================================================================================

#include <deal.II/lac/solver_cg.h>

template <int dim, int degree>
void customPDE<dim, degree>::solveIncrement(bool skip_time_dependent)
{

    // log time
    this->computing_timer.enter_subsection("matrixFreePDE: solveIncrements");
    Timer time;
    char buffer[200];

    // Set ChorinSwitch to false so steps 1 and 2 may occur
    ChorinSwitch = false;

    // Get the RHS of the explicit equations
    if (this->hasExplicitEquation && !skip_time_dependent) {
        this->computeExplicitRHS();
    }

    // solve for each field
    for (unsigned int fieldIndex = 0; fieldIndex < this->fields.size(); fieldIndex++) {
        this->currentFieldIndex = fieldIndex; // Used in computeLHS()

        // Add Neumann BC terms to the residual vector for the current field, if appropriate
        // Currently commented out because it isn't working yet
        // applyNeumannBCs();

        // Parabolic (first order derivatives in time) fields
        if (this->fields[fieldIndex].pdetype == EXPLICIT_TIME_DEPENDENT && !skip_time_dependent) {

            this->updateExplicitSolution(fieldIndex);

            // Set the Dirichelet values (hanging node constraints don't need to be distributed every time step, only at output)
            if (this->has_Dirichlet_BCs) {
                this->constraintsDirichletSet[fieldIndex]->distribute(*this->solutionSet[fieldIndex]);
            }

            this->solutionSet[fieldIndex]->update_ghost_values();

            // Print update to screen and confirm that solution isn't nan
            if (this->currentIncrement % userInputs.skip_print_steps == 0) {
                double solution_L2_norm = this->solutionSet[fieldIndex]->l2_norm();

                snprintf(buffer, sizeof(buffer), "field '%2s' [explicit solve]: current solution: %12.6e, current residual:%12.6e\n",
                    this->fields[fieldIndex].name.c_str(),
                    solution_L2_norm,
                    this->residualSet[fieldIndex]->l2_norm());
                this->pcout << buffer;

                if (!numbers::is_finite(solution_L2_norm)) {
                    snprintf(buffer, sizeof(buffer), "ERROR: field '%s' solution is NAN. exiting.\n\n",
                        this->fields[fieldIndex].name.c_str());
                    this->pcout << buffer;
                    exit(-1);
                }
            }
        }
    }

    // Now, update the non-explicit variables
    // For the time being, this is just the elliptic equations, but implicit parabolic and auxilary equations should also be here
    if (this->hasNonExplicitEquation && !skip_time_dependent) {

        bool nonlinear_it_converged = false;
        unsigned int nonlinear_it_index = 0;

        // Check to make sure that the first variable is the velocity vector
        if (userInputs.var_name[0] != "u") {
            std::cerr << "PRISMS-PF: Invalid field for 0. Must be the velocity field u." << std::endl;
            abort();
        }

        while (!nonlinear_it_converged) {
            nonlinear_it_converged = true; // Set to true here and will be set to false if any variable isn't converged

            // Update residualSet for the non-explicitly updated variables
            // Ideally, I'd just do this for the non-explicit variables, but for now I'll do all of them
            // this is a little redundant, but hopefully not too terrible
            this->computeNonexplicitRHS();

            for (unsigned int fieldIndex = 0; fieldIndex < this->fields.size(); fieldIndex++) {
                this->currentFieldIndex = fieldIndex; // Used in computeLHS()

                if ((this->fields[fieldIndex].pdetype == IMPLICIT_TIME_DEPENDENT && !skip_time_dependent) || this->fields[fieldIndex].pdetype == TIME_INDEPENDENT) {

                    if (this->currentIncrement % userInputs.skip_print_steps == 0 && userInputs.var_nonlinear[fieldIndex]) {
                        snprintf(buffer, sizeof(buffer), "field '%2s' [nonlinear solve]: current solution: %12.6e, current residual:%12.6e\n",
                            this->fields[fieldIndex].name.c_str(),
                            this->solutionSet[fieldIndex]->l2_norm(),
                            this->residualSet[fieldIndex]->l2_norm());
                        this->pcout << buffer;
                    }

                    // apply Dirichlet BC's
                    //  This clears the residual where we want to apply Dirichlet BCs, otherwise the solver sees a positive residual
                    this->constraintsDirichletSet[fieldIndex]->set_zero(*this->residualSet[fieldIndex]);

                    // solver controls
                    double tol_value;
                    if (MatrixFreePDE<dim, degree>::userInputs.linear_solver_parameters.getToleranceType(fieldIndex) == ABSOLUTE_RESIDUAL) {
                        tol_value = MatrixFreePDE<dim, degree>::userInputs.linear_solver_parameters.getToleranceValue(fieldIndex);
                    } else {
                        tol_value = MatrixFreePDE<dim, degree>::userInputs.linear_solver_parameters.getToleranceValue(fieldIndex) * this->residualSet[fieldIndex]->l2_norm();
                    }

                    SolverControl solver_control(MatrixFreePDE<dim, degree>::userInputs.linear_solver_parameters.getMaxIterations(fieldIndex), tol_value);

                    // Currently the only allowed solver is SolverCG, the SolverType input variable is a dummy
                    SolverCG<vectorType> solver(solver_control);

                    // solve
                    try {
                        if (this->fields[fieldIndex].type == SCALAR) {
                            this->dU_scalar = 0.0;
                            solver.solve(*this, this->dU_scalar, *this->residualSet[fieldIndex], IdentityMatrix(this->solutionSet[fieldIndex]->size()));
                        } else {
                            this->dU_vector = 0.0;
                            solver.solve(*this, this->dU_vector, *this->residualSet[fieldIndex], IdentityMatrix(this->solutionSet[fieldIndex]->size()));
                        }
                    } catch (...) {
                        this->pcout << "\nWarning: linear solver did not converge as per set tolerances. consider increasing the maximum number of iterations or decreasing the solver tolerance.\n";
                    }

                    if (userInputs.var_nonlinear[fieldIndex]) {

                        // Now that we have the calculated change in the solution, we need to select a damping coefficient
                        double damping_coefficient;

                        if (MatrixFreePDE<dim, degree>::userInputs.nonlinear_solver_parameters.getBacktrackDampingFlag(fieldIndex)) {
                            vectorType solutionSet_old = *this->solutionSet[fieldIndex];
                            double residual_old = this->residualSet[fieldIndex]->l2_norm();

                            damping_coefficient = 1.0;
                            bool damping_coefficient_found = false;
                            while (!damping_coefficient_found) {
                                if (this->fields[fieldIndex].type == SCALAR) {
                                    this->solutionSet[fieldIndex]->sadd(1.0, damping_coefficient, this->dU_scalar);
                                } else {
                                    this->solutionSet[fieldIndex]->sadd(1.0, damping_coefficient, this->dU_vector);
                                }

                                this->computeNonexplicitRHS();

                                for (std::map<types::global_dof_index, double>::const_iterator it = this->valuesDirichletSet[fieldIndex]->begin(); it != this->valuesDirichletSet[fieldIndex]->end(); ++it) {
                                    if (this->residualSet[fieldIndex]->in_local_range(it->first)) {
                                        (*this->residualSet[fieldIndex])(it->first) = 0.0;
                                    }
                                }

                                double residual_new = this->residualSet[fieldIndex]->l2_norm();

                                if (this->currentIncrement % userInputs.skip_print_steps == 0) {
                                    this->pcout << "    Old residual: " << residual_old << " Damping Coeff: " << damping_coefficient << " New Residual: " << residual_new << std::endl;
                                }

                                // An improved approach would use the Armijo–Goldstein condition to ensure a sufficent decrease in the residual. This way is just scales the residual.
                                if ((residual_new < (residual_old * MatrixFreePDE<dim, degree>::userInputs.nonlinear_solver_parameters.getBacktrackResidualDecreaseCoeff(fieldIndex))) || damping_coefficient < 1.0e-4) {
                                    damping_coefficient_found = true;
                                } else {
                                    damping_coefficient *= MatrixFreePDE<dim, degree>::userInputs.nonlinear_solver_parameters.getBacktrackStepModifier(fieldIndex);
                                    *this->solutionSet[fieldIndex] = solutionSet_old;
                                }
                            }
                        } else {
                            damping_coefficient = MatrixFreePDE<dim, degree>::userInputs.nonlinear_solver_parameters.getDefaultDampingCoefficient(fieldIndex);

                            if (this->fields[fieldIndex].type == SCALAR) {
                                this->solutionSet[fieldIndex]->sadd(1.0, damping_coefficient, this->dU_scalar);
                            } else {
                                this->solutionSet[fieldIndex]->sadd(1.0, damping_coefficient, this->dU_vector);
                            }
                        }

                        if (this->currentIncrement % userInputs.skip_print_steps == 0) {
                            double dU_norm;
                            if (this->fields[fieldIndex].type == SCALAR) {
                                dU_norm = this->dU_scalar.l2_norm();
                            } else {
                                dU_norm = this->dU_vector.l2_norm();
                            }
                            snprintf(buffer, sizeof(buffer), "field '%2s' [linear solve]: initial residual:%12.6e, current residual:%12.6e, nsteps:%u, tolerance criterion:%12.6e, solution: %12.6e, dU: %12.6e\n",
                                this->fields[fieldIndex].name.c_str(),
                                this->residualSet[fieldIndex]->l2_norm(),
                                solver_control.last_value(),
                                solver_control.last_step(), solver_control.tolerance(), this->solutionSet[fieldIndex]->l2_norm(), dU_norm);
                            this->pcout << buffer;
                        }

                        // Check to see if this individual variable has converged
                        if (MatrixFreePDE<dim, degree>::userInputs.nonlinear_solver_parameters.getToleranceType(fieldIndex) == ABSOLUTE_SOLUTION_CHANGE) {
                            double diff;

                            if (this->fields[fieldIndex].type == SCALAR) {
                                diff = this->dU_scalar.l2_norm();
                            } else {
                                diff = this->dU_vector.l2_norm();
                            }
                            if (this->currentIncrement % userInputs.skip_print_steps == 0) {
                                this->pcout << "Relative difference between nonlinear iterations: " << diff << " " << nonlinear_it_index << " " << this->currentIncrement << std::endl;
                            }

                            if (diff > MatrixFreePDE<dim, degree>::userInputs.nonlinear_solver_parameters.getToleranceValue(fieldIndex) && nonlinear_it_index < MatrixFreePDE<dim, degree>::userInputs.nonlinear_solver_parameters.getMaxIterations()) {
                                nonlinear_it_converged = false;
                            }
                        } else {
                            std::cerr << "PRISMS-PF Error: Nonlinear solver tolerance types other than ABSOLUTE_CHANGE have yet to be implemented." << std::endl;
                        }
                    } else {
                        if (nonlinear_it_index == 0) {

                            if (this->fields[fieldIndex].type == SCALAR) {
                                *this->solutionSet[fieldIndex] += this->dU_scalar;
                            } else {
                                *this->solutionSet[fieldIndex] += this->dU_vector;
                            }

                            if (this->currentIncrement % userInputs.skip_print_steps == 0) {
                                double dU_norm;
                                if (this->fields[fieldIndex].type == SCALAR) {
                                    dU_norm = this->dU_scalar.l2_norm();
                                } else {
                                    dU_norm = this->dU_vector.l2_norm();
                                }
                                snprintf(buffer, sizeof(buffer), "field '%2s' [linear solve]: initial residual:%12.6e, current residual:%12.6e, nsteps:%u, tolerance criterion:%12.6e, solution: %12.6e, dU: %12.6e\n",
                                    this->fields[fieldIndex].name.c_str(),
                                    this->residualSet[fieldIndex]->l2_norm(),
                                    solver_control.last_value(),
                                    solver_control.last_step(), solver_control.tolerance(), this->solutionSet[fieldIndex]->l2_norm(), dU_norm);
                                this->pcout << buffer;
                            }
                        }
                    }
                } else if (this->fields[fieldIndex].pdetype == AUXILIARY) {

                    if (userInputs.var_nonlinear[fieldIndex] || nonlinear_it_index == 0) {

                        // If the equation for this field is nonlinear, save the old solution
                        if (userInputs.var_nonlinear[fieldIndex]) {
                            if (this->fields[fieldIndex].type == SCALAR) {
                                this->dU_scalar = *this->solutionSet[fieldIndex];
                            } else {
                                this->dU_vector = *this->solutionSet[fieldIndex];
                            }
                        }

                        this->updateExplicitSolution(fieldIndex);

                        // Set the Dirichelet values (hanging node constraints don't need to be distributed every time step, only at output)
                        if (this->has_Dirichlet_BCs) {
                            this->constraintsDirichletSet[fieldIndex]->distribute(*this->solutionSet[fieldIndex]);
                        }

                        this->solutionSet[fieldIndex]->update_ghost_values();

                        // Print update to screen
                        if (this->currentIncrement % userInputs.skip_print_steps == 0) {
                            snprintf(buffer, sizeof(buffer), "field '%2s' [auxiliary solve]: current solution: %12.6e, current residual:%12.6e\n",
                                this->fields[fieldIndex].name.c_str(),
                                this->solutionSet[fieldIndex]->l2_norm(),
                                this->residualSet[fieldIndex]->l2_norm());
                            this->pcout << buffer;
                        }

                        // Check to see if this individual variable has converged
                        if (userInputs.var_nonlinear[fieldIndex]) {
                            if (MatrixFreePDE<dim, degree>::userInputs.nonlinear_solver_parameters.getToleranceType(fieldIndex) == ABSOLUTE_SOLUTION_CHANGE) {

                                double diff;

                                if (this->fields[fieldIndex].type == SCALAR) {
                                    this->dU_scalar -= *this->solutionSet[fieldIndex];
                                    diff = this->dU_scalar.l2_norm();
                                } else {
                                    this->dU_vector -= *this->solutionSet[fieldIndex];
                                    diff = this->dU_vector.l2_norm();
                                }
                                if (this->currentIncrement % userInputs.skip_print_steps == 0) {
                                    this->pcout << "Relative difference between nonlinear iterations: " << diff << " " << nonlinear_it_index << " " << this->currentIncrement << std::endl;
                                }

                                if (diff > MatrixFreePDE<dim, degree>::userInputs.nonlinear_solver_parameters.getToleranceValue(fieldIndex) && nonlinear_it_index < MatrixFreePDE<dim, degree>::userInputs.nonlinear_solver_parameters.getMaxIterations()) {
                                    nonlinear_it_converged = false;
                                }

                            } else {
                                std::cerr << "PRISMS-PF Error: Nonlinear solver tolerance types other than ABSOLUTE_CHANGE have yet to be implemented." << std::endl;
                            }
                        }
                    }
                }

                // check if solution is nan
                if (!numbers::is_finite(this->solutionSet[fieldIndex]->l2_norm()) && this->fields[fieldIndex].pdetype != EXPLICIT_TIME_DEPENDENT) {
                    snprintf(buffer, sizeof(buffer), "ERROR: field '%s' solution is NAN. exiting.\n\n",
                        this->fields[fieldIndex].name.c_str());
                    this->pcout << buffer;
                    exit(-1);
                }
            }

            nonlinear_it_index++;
        }
    }

    // Special methods for pressure correction in Chorin projection method
    unsigned int fieldIndex = 0;
    this->currentFieldIndex = fieldIndex; // Used in computeLHS()

    // Parabolic (first order derivatives in time) fields
    if (this->fields[fieldIndex].pdetype == EXPLICIT_TIME_DEPENDENT && userInputs.var_name[fieldIndex] == "u" && !skip_time_dependent) {

        // Set ChorinSwitch to true so steps 3 may occur
        ChorinSwitch = true;

        // Get the RHS of the new explicit equations
        this->computeExplicitRHS();

        // Set ChorinSwitch to false so steps 1 and 2 may occur
        ChorinSwitch = false;

        this->updateExplicitSolution(fieldIndex);

        // Set the Dirichlet values
        if (this->has_Dirichlet_BCs) {
            this->constraintsDirichletSet[fieldIndex]->distribute(*this->solutionSet[fieldIndex]);
        }
        // computing_timer.enter_subsection("matrixFreePDE: updateExplicitGhosts");
        this->solutionSet[fieldIndex]->update_ghost_values();
        // computing_timer.exit_subsection("matrixFreePDE: updateExplicitGhosts");

        // Print update to screen and confirm that solution isn't nan
        if (this->currentIncrement % userInputs.skip_print_steps == 0) {
            double solution_L2_norm = this->solutionSet[fieldIndex]->l2_norm();

            snprintf(buffer, sizeof(buffer), "field '%2s' [explicit solve]: current solution: %12.6e, current residual:%12.6e\n",
                this->fields[fieldIndex].name.c_str(),
                solution_L2_norm,
                this->residualSet[fieldIndex]->l2_norm());
            this->pcout << buffer;

            if (!numbers::is_finite(solution_L2_norm)) {
                snprintf(buffer, sizeof(buffer), "ERROR: field '%s' solution is NAN. exiting.\n\n",
                    this->fields[fieldIndex].name.c_str());
                this->pcout << buffer;
                exit(-1);
            }
        }
    }

    if (this->currentIncrement % userInputs.skip_print_steps == 0) {
        this->pcout << "wall time: " << time.wall_time() << "s\n";
    }
    // log time
    this->computing_timer.leave_subsection("matrixFreePDE: solveIncrements");
}