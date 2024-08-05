#include <deal.II/grid/grid_tools.h>

#include "../../include/matrixFreePDE.h"

template <int dim, int degree>
class customPDE : public MatrixFreePDE<dim, degree> {
public:
    // Constructor
    customPDE(userInputParameters<dim> _userInputs)
        : MatrixFreePDE<dim, degree>(_userInputs)
        , userInputs(_userInputs) {};

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

    double Re = userInputs.get_model_constant_double("Re");

    // This bool acts as a switch to indicate what Chorin projection step is being calculating
    bool ChorinSwitch = false;

    // ================================================================
};

// =================================================================================
// Function overriding solveIncrement
// =================================================================================

#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/solver_cg.h>

template <int dim, int degree>
void customPDE<dim, degree>::solveIncrement(bool skip_time_dependent)
{

    // log time
    this->computing_timer.enter_subsection("matrixFreePDE: solveIncrements");
    Timer time;

    // Set ChorinSwitch to false so steps 1 and 2 may occur
    ChorinSwitch = false;

    // Check if there is at least one explicit equation. If not, skip ahead
    if (!this->hasExplicitEquation) {
        goto nonexplicit;
    }
    // Check if skipping time dependent solves (e.g. initial condition). If so, skip ahead
    if (skip_time_dependent) {
        goto nonexplicit;
    }

    // Get the RHS of the explicit equations
    this->computeExplicitRHS();

    // solve for each field
    for (unsigned int fieldIndex = 0; fieldIndex < this->fields.size(); fieldIndex++) {

        // Parabolic (first order derivatives in time) fields
        if (this->fields[fieldIndex].pdetype == EXPLICIT_TIME_DEPENDENT) {

            // Explicit-time step each DOF
            this->updateExplicitSolution(fieldIndex);

            // Apply Boundary conditions
            this->applyBCs(fieldIndex);

            // Print update to screen and confirm that solution isn't nan
            if (this->tStep.currentIncrement % userInputs.skip_print_steps == 0) {
                this->printOutputs(fieldIndex);
            }
        }
    }

nonexplicit:
    // Check if there is at least one nonexplicit equation. If not, skip ahead
    if (!this->hasNonExplicitEquation) {
        goto end;
    }

    // Check to make sure that the first variable is the velocity vector
    if (userInputs.var_name[0] != "u") {
        std::cerr << "PRISMS-PF: Invalid field for 0. Must be the velocity field u." << std::endl;
        abort();
    }

    // Now, update the non-explicit variables
    for (unsigned int fieldIndex = 0; fieldIndex < this->fields.size(); fieldIndex++) {
        this->currentFieldIndex = fieldIndex; // Used in computeLHS()

        // Update residualSet for the non-explicitly updated variables
        this->computeNonexplicitRHS();

        if ((this->fields[fieldIndex].pdetype == IMPLICIT_TIME_DEPENDENT && !skip_time_dependent) || this->fields[fieldIndex].pdetype == TIME_INDEPENDENT) {
            bool nonlinear_it_converged = false;
            unsigned int nonlinear_it_index = 0;

            while (!nonlinear_it_converged) {
                // Update residualSet for the non-explicitly updated variables
                this->computeNonexplicitRHS();

                if (this->tStep.currentIncrement % userInputs.skip_print_steps == 0 && userInputs.var_nonlinear[fieldIndex]) {
                    this->printOutputs(fieldIndex);
                }

                // This clears the residual where we want to apply Dirichlet BCs, otherwise the solver sees a positive residual
                this->BCs.constraintsDirichletSet[fieldIndex]->set_zero(*this->residualSet[fieldIndex]);

                // Solve
                nonlinear_it_converged = this->nonlinearSolve(fieldIndex, nonlinear_it_index);

                // Apply Boundary conditions
                this->applyBCs(fieldIndex);

                nonlinear_it_index++;
            }
        } else if (this->fields[fieldIndex].pdetype == AUXILIARY) {

            this->updateExplicitSolution(fieldIndex);

            // Apply Boundary conditions
            this->applyBCs(fieldIndex);

            // Print update to screen
            if (this->tStep.currentIncrement % userInputs.skip_print_steps == 0) {
                this->printOutputs(fieldIndex);
            }
        }
    }

    // Special methods for pressure correction in Chorin projection method
    if (skip_time_dependent) {
        goto end;
    }

    // Set ChorinSwitch to true so steps 3 may occur
    ChorinSwitch = true;

    // Get the RHS of the new explicit equations
    this->computeExplicitRHS();

    // Set ChorinSwitch to false so steps 1 and 2 may occur
    ChorinSwitch = false;     

    // solve for the projected velocity field
    for (unsigned int fieldIndex = 0; fieldIndex < this->fields.size(); fieldIndex++) {

        // Here are the allowed fields that we recalulate
        bool skipLoop = true;
        if (userInputs.var_name[fieldIndex] == "u") {
            skipLoop = false;
        }
        if (skipLoop) {
            continue;
        }

        // Parabolic (first order derivatives in time) fields
        if (this->fields[fieldIndex].pdetype == EXPLICIT_TIME_DEPENDENT) {

            // Explicit-time step each DOF
            this->updateExplicitSolution(fieldIndex);

            // Apply Boundary conditions
            this->applyBCs(fieldIndex);

            // Print update to screen and confirm that solution isn't nan
            if (this->tStep.currentIncrement % userInputs.skip_print_steps == 0) {
                this->printOutputs(fieldIndex);
            }
        }
    }

end:
    if (this->tStep.currentIncrement % userInputs.skip_print_steps == 0) {
        this->pcout << "wall time: " << time.wall_time() << "s\n";
    }
    // log time
    this->computing_timer.leave_subsection("matrixFreePDE: solveIncrements");
}