// init() method for MatrixFreePDE class

#include <deal.II/grid/grid_generator.h>

#include "../../include/matrixFreePDE.h"
#include "../../include/varBCs.h"

// populate with fields and setup matrix free system
template <int dim, int degree>
void
MatrixFreePDE<dim, degree>::init()
{
  computing_timer.enter_subsection("matrixFreePDE: initialization");

  pcout << "creating problem mesh...\n";
  // Create the coarse mesh and mark the boundaries
  makeTriangulation(triangulation);

  // Set which (if any) faces of the triangulation are periodic
  setPeriodicity();

  // If resuming from a checkpoint, load the refined triangulation, otherwise refine
  // globally per the parameters.in file
  userInputs.resume_from_checkpoint
    ? load_checkpoint_triangulation()
    : triangulation.refine_global(userInputs.refine_factor);

  // Write out the size of the computational domain and the total number of elements
  pcout << "problem dimensions: " << userInputs.domain_size[0] << "x"
        << userInputs.domain_size[1];
  if (dim == 3)
    {
      pcout << "x" << userInputs.domain_size[2];
    }
  pcout << std::endl
        << "number of elements: " << triangulation.n_global_active_cells() << std::endl
        << std::endl;

  // Setup system
  pcout << "initializing matrix free object\n";
  n_dofs = 0;
  for (auto &field : fields)
    {
      std::string var_type;
      switch (field.pdetype)
        {
          case (EXPLICIT_TIME_DEPENDENT):
            var_type            = "EXPLICIT_TIME_DEPENDENT";
            isTimeDependentBVP  = true;
            hasExplicitEquation = true;
            break;
          case (IMPLICIT_TIME_DEPENDENT):
            var_type               = "IMPLICIT_TIME_DEPENDENT";
            isTimeDependentBVP     = true;
            hasNonExplicitEquation = true;
            break;
          case (TIME_INDEPENDENT):
            var_type               = "TIME_INDEPENDENT";
            isEllipticBVP          = true;
            hasNonExplicitEquation = true;
            break;
          case (AUXILIARY):
            var_type               = "AUXILIARY";
            hasNonExplicitEquation = true;
            break;
        }

      currentFieldIndex = field.index;

      char buffer[100];

      snprintf(buffer,
               sizeof(buffer),
               "initializing finite element space P^%u for %9s:%6s field '%s'\n",
               degree,
               var_type.c_str(),
               (field.type == SCALAR ? "SCALAR" : "VECTOR"),
               field.name.c_str());
      pcout << buffer;

      // create FESystem
      Assert(field.type == SCALAR || field.type == VECTOR,
             ExcMessage("PRISMS-PF Error: Unknown field type. The only allowed fields "
                        "are scalar / vector."));

      if (field.type == SCALAR)
        {
          FE_set.push_back(
            std::make_unique<FESystem<dim>>(FE_Q<dim>(QGaussLobatto<1>(degree + 1)), 1));
        }
      else if (field.type == VECTOR)
        {
          FE_set.push_back(
            std::make_unique<FESystem<dim>>(FE_Q<dim>(QGaussLobatto<1>(degree + 1)),
                                            dim));
        }

      // distribute DOFs
      DoFHandler<dim> *dof_handler;

      dof_handler = new DoFHandler<dim>(triangulation);
      dof_handler_set.push_back(dof_handler);
      dof_handler_set_nonconst.push_back(dof_handler);

      dof_handler->distribute_dofs(*FE_set.back());
      n_dofs += dof_handler->n_dofs();

      // Extract locally_relevant_dofs
      IndexSet *locally_relevant_dofs;

      locally_relevant_dofs = new IndexSet;
      locally_relevant_dofsSet.push_back(locally_relevant_dofs);
      locally_relevant_dofsSet_nonconst.push_back(locally_relevant_dofs);

      locally_relevant_dofs->clear();
      DoFTools::extract_locally_relevant_dofs(*dof_handler, *locally_relevant_dofs);

      // Create constraints
      AffineConstraints<double> *constraintsDirichlet, *constraintsOther;

      constraintsDirichlet = new AffineConstraints<double>;
      constraintsDirichletSet.push_back(constraintsDirichlet);
      constraintsDirichletSet_nonconst.push_back(constraintsDirichlet);
      constraintsOther = new AffineConstraints<double>;
      constraintsOtherSet.push_back(constraintsOther);
      constraintsOtherSet_nonconst.push_back(constraintsOther);
      valuesDirichletSet.push_back(new std::map<dealii::types::global_dof_index, double>);

      constraintsDirichlet->clear();
      constraintsDirichlet->reinit(*locally_relevant_dofs);
      constraintsOther->clear();
      constraintsOther->reinit(*locally_relevant_dofs);

      // Get hanging node constraints
      DoFTools::make_hanging_node_constraints(*dof_handler, *constraintsOther);

      // Add a constraint to fix the value at the origin to zero if all BCs are
      // zero-derivative or periodic
      std::vector<int> rigidBodyModeComponents;
      // getComponentsWithRigidBodyModes(rigidBodyModeComponents);
      // setRigidBodyModeConstraints(rigidBodyModeComponents,constraintsOther,dof_handler);

      // Get constraints for periodic BCs
      setPeriodicityConstraints(constraintsOther, dof_handler);

      // Check if Dirichlet BCs are used
      for (unsigned int i = 0; i < userInputs.BC_list.size(); i++)
        {
          for (unsigned int direction = 0; direction < 2 * dim; direction++)
            {
              if (userInputs.BC_list[i].var_BC_type[direction] == DIRICHLET)
                {
                  field.hasDirichletBCs = true;
                }
              else if (userInputs.BC_list[i].var_BC_type[direction] ==
                       NON_UNIFORM_DIRICHLET)
                {
                  field.hasnonuniformDirichletBCs = true;
                }
              else if (userInputs.BC_list[i].var_BC_type[direction] == NEUMANN)
                {
                  field.hasNeumannBCs = true;
                }
            }
        }

      // Get constraints for Dirichlet BCs
      applyDirichletBCs();

      constraintsDirichlet->close();
      constraintsOther->close();

      // Store Dirichlet BC DOF's
      valuesDirichletSet[field.index]->clear();
      for (types::global_dof_index i = 0; i < dof_handler->n_dofs(); i++)
        {
          if (locally_relevant_dofs->is_element(i))
            {
              if (constraintsDirichlet->is_constrained(i))
                {
                  (*valuesDirichletSet[field.index])[i] =
                    constraintsDirichlet->get_inhomogeneity(i);
                }
            }
        }

      snprintf(buffer,
               sizeof(buffer),
               "field '%2s' DOF : %u (Constraint DOF : %u)\n",
               field.name.c_str(),
               dof_handler->n_dofs(),
               constraintsDirichlet->n_constraints());
      pcout << buffer;
    }
  pcout << "total DOF : " << n_dofs << std::endl;

  // Setup the matrix free object
  typename MatrixFree<dim, double>::AdditionalData additional_data;
// The member "mpi_communicator" was removed in deal.II version 8.5 but is
// required before it
#if (DEAL_II_VERSION_MAJOR < 9 && DEAL_II_VERSION_MINOR < 5)
  additional_data.mpi_communicator = MPI_COMM_WORLD;
#endif
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, double>::AdditionalData::partition_partition;
  additional_data.mapping_update_flags =
    (update_values | update_gradients | update_JxW_values | update_quadrature_points);
  QGaussLobatto<1> quadrature(degree + 1);
  matrixFreeObject.clear();
#if (DEAL_II_VERSION_MAJOR == 9 && DEAL_II_VERSION_MINOR < 4)
  matrixFreeObject.reinit(dof_handler_set,
                          constraintsOtherSet,
                          quadrature,
                          additional_data);
#else
  matrixFreeObject.reinit(MappingFE<dim, dim>(FE_Q<dim>(QGaussLobatto<1>(degree + 1))),
                          dof_handler_set,
                          constraintsOtherSet,
                          quadrature,
                          additional_data);
#endif
  bool dU_scalar_init = false;
  bool dU_vector_init = false;

  // Setup solution vectors
  pcout << "initializing parallel::distributed residual and solution vectors\n";

  solution_set.reserve(fields.size());
  residual_set.reserve(fields.size());

  for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++)
    {
      vectorType *U = new vectorType;
      vectorType *R = new vectorType;

      matrixFreeObject.initialize_dof_vector(*R, fieldIndex);
      matrixFreeObject.initialize_dof_vector(*U, fieldIndex);

      solution_set.push_back(U);
      residual_set.push_back(R);

      *R = 0;
      *U = 0;

      // Initializing temporary dU vector required for implicit solves of the
      // elliptic equation.
      if (fields[fieldIndex].pdetype == TIME_INDEPENDENT ||
          fields[fieldIndex].pdetype == IMPLICIT_TIME_DEPENDENT ||
          (fields[fieldIndex].pdetype == AUXILIARY &&
           userInputs.var_nonlinear[fieldIndex]))
        {
          if (fields[fieldIndex].type == SCALAR && !dU_scalar_init)
            {
              matrixFreeObject.initialize_dof_vector(dU_scalar, fieldIndex);
              dU_scalar_init = true;
            }
          else if (fields[fieldIndex].type == SCALAR && !dU_vector_init)
            {
              matrixFreeObject.initialize_dof_vector(dU_vector, fieldIndex);
              dU_vector_init = true;
            }
        }
    }

  // check if time dependent BVP and compute invM
  if (isTimeDependentBVP)
    {
      computeInvM();
    }

  // Apply the initial conditions to the solution vectors
  // The initial conditions are re-applied below in the "do_adaptive_refinement"
  // function so that the mesh can adapt based on the initial conditions.
  if (userInputs.resume_from_checkpoint)
    {
      load_checkpoint_fields();
    }
  else
    {
      applyInitialConditions();
    }

  // Create new solution transfer sets (needed for the "refine_grid" call, might
  // be able to move this elsewhere)
  solution_transfer_set.clear();
  for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++)
    {
      solution_transfer_set.push_back(
        new parallel::distributed::SolutionTransfer<dim, vectorType>(
          *dof_handler_set_nonconst[fieldIndex]));
    }

  // Ghost the solution vectors. Also apply the constraints (if any) on the
  // solution vectors
  for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++)
    {
      constraintsDirichletSet[fieldIndex]->distribute(*solution_set[fieldIndex]);
      constraintsOtherSet[fieldIndex]->distribute(*solution_set[fieldIndex]);
      solution_set[fieldIndex]->update_ghost_values();
    }

  // If not resuming from a checkpoint, check and perform adaptive mesh refinement, which
  // reinitializes the system with the new mesh
  if (!userInputs.resume_from_checkpoint && userInputs.h_adaptivity == true)
    {
      computing_timer.enter_subsection("matrixFreePDE: AMR");

      unsigned int numDoF_preremesh = n_dofs;
      for (unsigned int remesh_index = 0;
           remesh_index <
           (userInputs.max_refinement_level - userInputs.min_refinement_level);
           remesh_index++)
        {
          AMR.do_adaptive_refinement(currentIncrement);
          reinit();
          if (n_dofs == numDoF_preremesh)
            break;
          numDoF_preremesh = n_dofs;
        }

      computing_timer.leave_subsection("matrixFreePDE: AMR");
    }

  // If resuming from a checkpoint, load the proper starting increment and time
  if (userInputs.resume_from_checkpoint)
    {
      load_checkpoint_time_info();
    }

  computing_timer.leave_subsection("matrixFreePDE: initialization");
}

template <int dim, int degree>
void
MatrixFreePDE<dim, degree>::makeTriangulation(
  parallel::distributed::Triangulation<dim> &tria) const
{
  Point<dim> origin;
  Point<dim> upper_corner;

  for (unsigned int i = 0; i < dim; ++i)
    {
      upper_corner[i] = userInputs.domain_size[i];
    }

  GridGenerator::subdivided_hyper_rectangle(tria,
                                            userInputs.subdivisions,
                                            origin,
                                            upper_corner);

  // Mark boundaries for applying the boundary conditions
  markBoundaries(tria);
}

#include "../../include/matrixFreePDE_template_instantiations.h"
