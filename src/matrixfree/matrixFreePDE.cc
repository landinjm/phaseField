// constructor and destructor for matrixFreePDE class

#include "../../include/matrixFreePDE.h"

// constructor
template <int dim, int degree>
MatrixFreePDE<dim, degree>::MatrixFreePDE(userInputParameters<dim> _userInputs)
  : Subscriptor()
  , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  , userInputs(_userInputs)
  , triangulation(MPI_COMM_WORLD)
  , currentFieldIndex(0)
  , isTimeDependentBVP(false)
  , isEllipticBVP(false)
  , hasExplicitEquation(false)
  , hasNonExplicitEquation(false)
  , currentTime(0.0)
  , currentIncrement(0)
  , currentOutput(0)
  , currentCheckpoint(0)
  , current_grain_reassignment(0)
  , computing_timer(pcout, TimerOutput::summary, TimerOutput::wall_times)
  , first_integrated_var_output_complete(false)
  , AMR(_userInputs,
        triangulation,
        fields,
        solution_set,
        solution_transfer_set,
        FE_set,
        dof_handler_set_nonconst,
        constraintsDirichletSet,
        constraintsOtherSet)
{}

// destructor
template <int dim, int degree>
MatrixFreePDE<dim, degree>::~MatrixFreePDE()
{
  matrixFreeObject.clear();

  // Delete the pointers contained in several member variable vectors
  // The size of each of these must be checked individually in case an exception
  // is thrown as they are being initialized.
  for (unsigned int iter = 0; iter < locally_relevant_dofsSet.size(); iter++)
    {
      delete locally_relevant_dofsSet[iter];
    }
  for (unsigned int iter = 0; iter < constraintsDirichletSet.size(); iter++)
    {
      delete constraintsDirichletSet[iter];
    }
  for (unsigned int iter = 0; iter < solution_transfer_set.size(); iter++)
    {
      delete solution_transfer_set[iter];
    }
  for (unsigned int iter = 0; iter < dof_handler_set.size(); iter++)
    {
      delete dof_handler_set[iter];
    }
  for (unsigned int iter = 0; iter < solution_set.size(); iter++)
    {
      delete solution_set[iter];
    }
  for (unsigned int iter = 0; iter < residual_set.size(); iter++)
    {
      delete residual_set[iter];
    }
}

#include "../../include/matrixFreePDE_template_instantiations.h"
