// constructor and destructor for matrixFreePDE class

#include "../../include/matrixFreePDE.h"

// constructor
template <int dim, int degree>
MatrixFreePDE<dim, degree>::MatrixFreePDE(userInputParameters<dim> _userInputs)
    : Subscriptor()
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , userInputs(_userInputs)
    , currentFieldIndex(0)
    , isTimeDependentBVP(false)
    , isEllipticBVP(false)
    , hasExplicitEquation(false)
    , hasNonExplicitEquation(false)
    , currentOutput(0)
    , current_grain_reassignment(0)
    , computing_timer(pcout, TimerOutput::summary, TimerOutput::wall_times)
    , first_integrated_var_output_complete(false)
    , Discretization(_userInputs)
    , BCs(_userInputs, Discretization)
    , RefineAdaptively(_userInputs, Discretization, BCs, fields, tStep.solutionSet, soltransSet)
    , tStep(_userInputs)
    , checkpoints(_userInputs, Discretization, tStep)
{
}

// destructor
template <int dim, int degree>
MatrixFreePDE<dim, degree>::~MatrixFreePDE()
{
    Discretization.matrixFreeObject.clear();

    // Delete the pointers contained in several member variable vectors
    // The size of each of these must be checked individually in case an exception is thrown
    // as they are being initialized.
    for (unsigned int iter = 0; iter < Discretization.locally_relevant_dofsSet.size(); iter++) {
        delete Discretization.locally_relevant_dofsSet[iter];
    }
    for (unsigned int iter = 0; iter < BCs.constraintsDirichletSet.size(); iter++) {
        delete BCs.constraintsDirichletSet[iter];
    }
    for (unsigned int iter = 0; iter < soltransSet.size(); iter++) {
        delete soltransSet[iter];
    }
    for (unsigned int iter = 0; iter < Discretization.dofHandlersSet.size(); iter++) {
        delete Discretization.dofHandlersSet[iter];
    }
    for (unsigned int iter = 0; iter < Discretization.FESet.size(); iter++) {
        delete Discretization.FESet[iter];
    }
    for (unsigned int iter = 0; iter < tStep.solutionSet.size(); iter++) {
        delete tStep.solutionSet[iter];
    }
    for (unsigned int iter = 0; iter < residualSet.size(); iter++) {
        delete residualSet[iter];
    }
}

#include "../../include/matrixFreePDE_template_instantiations.h"
