// solveIncrement() method for MatrixFreePDE class

#include <deal.II/lac/solver_cg.h>

#include "../../include/matrixFreePDE.h"

// solve each time increment
template <int dim, int degree>
void MatrixFreePDE<dim, degree>::solveIncrement(bool skip_time_dependent)
{
    // log time
    computing_timer.enter_subsection("matrixFreePDE: solveIncrements");
    Timer time;

    // Check if there is at least one explicit equation. If not, skip ahead
    if (!hasExplicitEquation) {
        goto nonexplicit;
    }
    // Check if skipping time dependent solves (e.g. initial condition). If so, skip ahead
    if (skip_time_dependent) {
        goto nonexplicit;
    }

    // Get the RHS of the explicit equations
    computeExplicitRHS();

    // solve for each field
    for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++) {

        // Parabolic (first order derivatives in time) fields
        if (fields[fieldIndex].pdetype == EXPLICIT_TIME_DEPENDENT) {

            updateExplicitSolution(fieldIndex);

            // Apply Boundary conditions
            applyBCs(fieldIndex);

            // Print update to screen and confirm that solution isn't nan
            if (tStep.currentIncrement % userInputs.skip_print_steps == 0) {
                printOutputs(fieldIndex);
            }
        }
    }

    nonexplicit:
        // Check if there is at least one nonexplicit equation. If not, skip ahead
        if (!hasNonExplicitEquation) {
            goto end;
        }
        // Now, update the non-explicit variables
        for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++) {
        currentFieldIndex = fieldIndex; // Used in computeLHS()

            // Update residualSet for the non-explicitly updated variables
            computeNonexplicitRHS();

            if ((fields[fieldIndex].pdetype == IMPLICIT_TIME_DEPENDENT && !skip_time_dependent) || fields[fieldIndex].pdetype == TIME_INDEPENDENT) {
                bool nonlinear_it_converged = false;
                unsigned int nonlinear_it_index = 0;

                while (!nonlinear_it_converged) {
                    // Update residualSet for the non-explicitly updated variables
                    computeNonexplicitRHS();

                    if (tStep.currentIncrement % userInputs.skip_print_steps == 0 && userInputs.var_nonlinear[fieldIndex]) {
                        printOutputs(fieldIndex);
                    }

                    // This clears the residual where we want to apply Dirichlet BCs, otherwise the solver sees a positive residual
                    BCs.constraintsDirichletSet[fieldIndex]->set_zero(*residualSet[fieldIndex]);

                    // Solve
                    nonlinear_it_converged = nonlinearSolve(fieldIndex, nonlinear_it_index);

                    // Apply Boundary conditions
                    applyBCs(fieldIndex);

                    nonlinear_it_index++;
                }
            }
            else if (fields[fieldIndex].pdetype == AUXILIARY) {
                    
                updateExplicitSolution(fieldIndex);

                // Apply Boundary conditions
                applyBCs(fieldIndex);

                // Print update to screen
                if (tStep.currentIncrement % userInputs.skip_print_steps == 0) {
                    printOutputs(fieldIndex);
                }
            }
        }

    end:
        if (tStep.currentIncrement % userInputs.skip_print_steps == 0) {
            pcout << "wall time: " << time.wall_time() << "s\n";
        }
        // log time
        computing_timer.leave_subsection("matrixFreePDE: solveIncrements");
}

// Nonlinear solving
template <int dim, int degree>
bool MatrixFreePDE<dim, degree>::nonlinearSolve(unsigned int fieldIndex, unsigned int nonlinear_it_index){

    bool nonlinear_it_converged = true; // Set to true here and will be set to false if any variable isn't converged

    // solver controls
    double tol_value;
    if (userInputs.linear_solver_parameters.getToleranceType(fieldIndex) == ABSOLUTE_RESIDUAL) {
        tol_value = userInputs.linear_solver_parameters.getToleranceValue(fieldIndex);
    } else {
        tol_value = userInputs.linear_solver_parameters.getToleranceValue(fieldIndex) * residualSet[fieldIndex]->l2_norm();
    }

    SolverControl solver_control(userInputs.linear_solver_parameters.getMaxIterations(fieldIndex), tol_value);

    // Currently the only allowed solver is SolverCG, the SolverType input variable is a dummy
    SolverCG<vectorType> solver(solver_control);

    // solve
    try {
        if (fields[fieldIndex].type == SCALAR) {
            dU_scalar = 0.0;
            solver.solve(*this, dU_scalar, *residualSet[fieldIndex], IdentityMatrix(tStep.solutionSet[fieldIndex]->size()));
        } else {
            dU_vector = 0.0;
            solver.solve(*this, dU_vector, *residualSet[fieldIndex], IdentityMatrix(tStep.solutionSet[fieldIndex]->size()));
        }
    } catch (...) {
        pcout << "\nWarning: linear solver did not converge as per set tolerances. consider increasing the maximum number of iterations or decreasing the solver tolerance.\n";
    }

    if (userInputs.var_nonlinear[fieldIndex]) {
        // Now that we have the calculated change in the solution, we need to select a damping coefficient
        double damping_coefficient;

        if (userInputs.nonlinear_solver_parameters.getBacktrackDampingFlag(fieldIndex)) {
            vectorType solutionSet_old = *tStep.solutionSet[fieldIndex];
            double residual_old = residualSet[fieldIndex]->l2_norm();

            damping_coefficient = 1.0;
            bool damping_coefficient_found = false;
            while (!damping_coefficient_found) {
                if (fields[fieldIndex].type == SCALAR) {
                    tStep.solutionSet[fieldIndex]->sadd(1.0, damping_coefficient, dU_scalar);
                } else {
                    tStep.solutionSet[fieldIndex]->sadd(1.0, damping_coefficient, dU_vector);
                }

                computeNonexplicitRHS();

                BCs.constraintsDirichletSet[fieldIndex]->set_zero(*residualSet[fieldIndex]);

                double residual_new = residualSet[fieldIndex]->l2_norm();

                if (tStep.currentIncrement % userInputs.skip_print_steps == 0) {
                    pcout << "    Old residual: " << residual_old << " Damping Coeff: " << damping_coefficient << " New Residual: " << residual_new << std::endl;
                }

                // An improved approach would use the Armijo–Goldstein condition to ensure a sufficent decrease in the residual. This way is just scales the residual.
                if ((residual_new < (residual_old * userInputs.nonlinear_solver_parameters.getBacktrackResidualDecreaseCoeff(fieldIndex))) || damping_coefficient < 1.0e-4) {
                    damping_coefficient_found = true;
                } else {
                    damping_coefficient *= userInputs.nonlinear_solver_parameters.getBacktrackStepModifier(fieldIndex);
                    *tStep.solutionSet[fieldIndex] = solutionSet_old;
                }
            }
        } else {
            damping_coefficient = userInputs.nonlinear_solver_parameters.getDefaultDampingCoefficient(fieldIndex);

            if (fields[fieldIndex].type == SCALAR) {
                tStep.solutionSet[fieldIndex]->sadd(1.0, damping_coefficient, dU_scalar);
            } else {
                tStep.solutionSet[fieldIndex]->sadd(1.0, damping_coefficient, dU_vector);
            }
        }

        if (tStep.currentIncrement % userInputs.skip_print_steps == 0) {
            printOutputs(fieldIndex, &solver_control);
        }

        // Check to see if this individual variable has converged
        if (userInputs.nonlinear_solver_parameters.getToleranceType(fieldIndex) == ABSOLUTE_SOLUTION_CHANGE) {
            double diff;

            if (fields[fieldIndex].type == SCALAR) {
                diff = dU_scalar.l2_norm();
            } else {
                diff = dU_vector.l2_norm();
            }
            if (tStep.currentIncrement % userInputs.skip_print_steps == 0) {
                pcout << "Relative difference between nonlinear iterations: " << diff << " " << nonlinear_it_index << " " << tStep.currentIncrement << std::endl;
            }

            if (diff > userInputs.nonlinear_solver_parameters.getToleranceValue(fieldIndex) && nonlinear_it_index < userInputs.nonlinear_solver_parameters.getMaxIterations()) {
                nonlinear_it_converged = false;
            }
        } else {
            std::cerr << "PRISMS-PF Error: Nonlinear solver tolerance types other than ABSOLUTE_CHANGE have yet to be implemented." << std::endl;
        }
    } else {
        if (nonlinear_it_index == 0) {

            if (fields[fieldIndex].type == SCALAR) {
                *tStep.solutionSet[fieldIndex] += dU_scalar;
            } else {
                *tStep.solutionSet[fieldIndex] += dU_vector;
            }

            if (tStep.currentIncrement % userInputs.skip_print_steps == 0) {
                printOutputs(fieldIndex, &solver_control);
            }
        }
    }

    return nonlinear_it_converged;
}

// Print outputs of solution & residual set
template <int dim, int degree>
void MatrixFreePDE<dim, degree>::printOutputs(unsigned int fieldIndex, SolverControl *solver_control)
{
    //Character limit for output buffer
    char buffer[200];

    double solution_L2_norm = tStep.solutionSet[fieldIndex]->l2_norm();
    double residual_L2_norm = residualSet[fieldIndex]->l2_norm();

    if (fields[fieldIndex].pdetype == EXPLICIT_TIME_DEPENDENT) {
        snprintf(buffer, sizeof(buffer), "field '%2s' [explicit solve]: current solution: %12.6e, current residual:%12.6e\n",
            fields[fieldIndex].name.c_str(),
            solution_L2_norm,
            residual_L2_norm);
    }
    else if (fields[fieldIndex].pdetype == IMPLICIT_TIME_DEPENDENT || fields[fieldIndex].pdetype == TIME_INDEPENDENT) {
        if (userInputs.var_nonlinear[fieldIndex]) {
            snprintf(buffer, sizeof(buffer), "field '%2s' [nonlinear solve]: current solution: %12.6e, current residual:%12.6e\n",
                fields[fieldIndex].name.c_str(),
                solution_L2_norm,
                residual_L2_norm);
        }
        if (solver_control){
            double dU_norm;
            if (fields[fieldIndex].type == SCALAR) {
                dU_norm = dU_scalar.l2_norm();
            } else {
                dU_norm = dU_vector.l2_norm();
            }

            snprintf(buffer, sizeof(buffer), "field '%2s' [linear solve]: initial residual:%12.6e, current residual:%12.6e, nsteps:%u, tolerance criterion:%12.6e, solution: %12.6e, dU: %12.6e\n",
                fields[fieldIndex].name.c_str(),
                residual_L2_norm,
                solver_control->last_value(),
                solver_control->last_step(),
                solver_control->tolerance(),
                solution_L2_norm,
                dU_norm);
        }
    }
    else if (fields[fieldIndex].pdetype == AUXILIARY) {
        snprintf(buffer, sizeof(buffer), "field '%2s' [auxiliary solve]: current solution: %12.6e, current residual:%12.6e\n",
            fields[fieldIndex].name.c_str(),
            solution_L2_norm,
            residual_L2_norm);
    }

    pcout << buffer;

    if (!numbers::is_finite(solution_L2_norm)) {
        snprintf(buffer, sizeof(buffer), "ERROR: field '%s' solution is NAN. exiting.\n\n",
            fields[fieldIndex].name.c_str());
        pcout << buffer;
        exit(-1);
    }
}

// Application of boundary conditions
template <int dim, int degree>
void MatrixFreePDE<dim, degree>::applyBCs(unsigned int fieldIndex)
{
    // Add Neumann BCs
    if (fields[fieldIndex].hasNeumannBCs) {
        // Currently commented out because it isn't working yet
        // applyNeumannBCs();
    }

    // Set the Dirichelet values (hanging node constraints don't need to be distributed every time step, only at output)
    if (fields[fieldIndex].hasDirichletBCs) {

        // Apply non-uniform Dirlichlet_BCs to the current field
        if (fields[fieldIndex].hasnonuniformDirichletBCs) {
            DoFHandler<dim>* dof_handler;
            dof_handler = Discretization.dofHandlersSet_nonconst.at(currentFieldIndex);
            IndexSet* locally_relevant_dofs;
            locally_relevant_dofs = Discretization.locally_relevant_dofsSet_nonconst.at(currentFieldIndex);
            locally_relevant_dofs->clear();
            DoFTools::extract_locally_relevant_dofs(*dof_handler, *locally_relevant_dofs);
            AffineConstraints<double>* constraintsDirichlet;
            constraintsDirichlet = BCs.constraintsDirichletSet_nonconst.at(currentFieldIndex);
            constraintsDirichlet->clear();
            constraintsDirichlet->reinit(*locally_relevant_dofs);
            applyDirichletBCs();
            constraintsDirichlet->close();
        }
        // Distribute for Uniform or Non-Uniform Dirichlet BCs
        BCs.constraintsDirichletSet[fieldIndex]->distribute(*tStep.solutionSet[fieldIndex]);
    }
    tStep.solutionSet[fieldIndex]->update_ghost_values();
}

// Explicit time step for matrixfree solve
template <int dim, int degree>
void MatrixFreePDE<dim, degree>::updateExplicitSolution(unsigned int fieldIndex)
{
    // Explicit-time step each DOF
    // Takes advantage of knowledge that the length of solutionSet and residualSet is an integer multiple of the length of invM for vector variables
    if (fields[fieldIndex].type == SCALAR) {
#if (DEAL_II_VERSION_MAJOR == 9 && DEAL_II_VERSION_MINOR < 4)
        unsigned int invM_size = invMscalar.local_size();
        for (unsigned int dof = 0; dof < tStep.solutionSet[fieldIndex]->local_size(); ++dof) {
#else
        unsigned int invM_size = invMscalar.locally_owned_size();
        for (unsigned int dof = 0; dof < tStep.solutionSet[fieldIndex]->locally_owned_size(); ++dof) {
#endif
            tStep.solutionSet[fieldIndex]->local_element(dof) = invMscalar.local_element(dof % invM_size) * residualSet[fieldIndex]->local_element(dof);
        }
    } else if (fields[fieldIndex].type == VECTOR) {
#if (DEAL_II_VERSION_MAJOR == 9 && DEAL_II_VERSION_MINOR < 4)
        unsigned int invM_size = invMvector.local_size();
        for (unsigned int dof = 0; dof < tStep.solutionSet[fieldIndex]->local_size(); ++dof) {
#else
        unsigned int invM_size = invMvector.locally_owned_size();
        for (unsigned int dof = 0; dof < tStep.solutionSet[fieldIndex]->locally_owned_size(); ++dof) {
#endif
            tStep.solutionSet[fieldIndex]->local_element(dof) = invMvector.local_element(dof % invM_size) * residualSet[fieldIndex]->local_element(dof);
        }
    }
}

#include "../../include/matrixFreePDE_template_instantiations.h"
