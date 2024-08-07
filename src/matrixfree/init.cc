// init() method for MatrixFreePDE class

#include <deal.II/grid/grid_generator.h>

#include "../../include/matrixFreePDE.h"
#include "../../include/varBCs.h"

// populate with fields and setup matrix free system
template <int dim, int degree>
void MatrixFreePDE<dim, degree>::init()
{
    computing_timer.enter_subsection("matrixFreePDE: initialization");

    // creating mesh
    pcout << "creating problem mesh...\n";

    // Create the coarse mesh and mark the boundaries
    Discretization.makeTriangulation();

    // Set which (if any) faces of the triangulation are periodic
    BCs.setPeriodicity();

    // If resuming from a checkpoint, load the refined triangulation, otherwise refine globally per the parameters.prm file
    if (userInputs.resume_from_checkpoint) {
        checkpoints.load_checkpoint_triangulation();
    } else {
        // Do the initial global refinement
        Discretization.triangulation.refine_global(userInputs.refine_factor);
    }

    // Write out the size of the computational domain and the total number of elements
    if (dim < 3) {
        pcout << "problem dimensions: " << userInputs.domain_size[0] << "x" << userInputs.domain_size[1] << std::endl;
    } else {
        pcout << "problem dimensions: " << userInputs.domain_size[0] << "x" << userInputs.domain_size[1] << "x" << userInputs.domain_size[2] << std::endl;
    }
    pcout << "number of elements: " << Discretization.triangulation.n_global_active_cells() << std::endl << std::endl;

    // Setup system
    pcout << "initializing matrix free object\n";
    Discretization.totalDOFs = 0;
    for (typename std::vector<Field<dim>>::iterator it = fields.begin(); it != fields.end(); ++it) {
        currentFieldIndex = it->index;

        char buffer[100];

        std::string var_type;
        if (it->pdetype == EXPLICIT_TIME_DEPENDENT) {
            var_type = "EXPLICIT_TIME_DEPENDENT";
            pFlags.isTimeDependentBVP = true;
            pFlags.hasExplicitEquation = true;
            
        } else if (it->pdetype == IMPLICIT_TIME_DEPENDENT) {
            var_type = "IMPLICIT_TIME_DEPENDENT";
            pFlags.isTimeDependentBVP = true;
            pFlags.hasNonExplicitEquation = true;
            std::cerr << "PRISMS-PF Error: IMPLICIT_TIME_DEPENDENT equation types are not currently supported" << std::endl;
            abort();

        } else if (it->pdetype == TIME_INDEPENDENT) {
            var_type = "TIME_INDEPENDENT";
            pFlags.isEllipticBVP = true;
            pFlags.hasNonExplicitEquation = true;

        } else if (it->pdetype == AUXILIARY) {
            var_type = "AUXILIARY";
            pFlags.hasNonExplicitEquation = true;

        }

        snprintf(buffer, sizeof(buffer), "initializing finite element space P^%u for %9s:%6s field '%s'\n",
            degree,
            var_type.c_str(),
            (it->type == SCALAR ? "SCALAR" : "VECTOR"),
            it->name.c_str());
        pcout << buffer;

        // create FESystem
        Discretization.makeFESystem(it->type, degree);

        // distribute DOFs
        Discretization.makeDOFs(currentFieldIndex);

        // Extract locally_relevant_dofs
        Discretization.extractLocalDOFs(currentFieldIndex);

        // Create constraints
        BCs.makeDirichletConstraints(currentFieldIndex);
        RefineAdaptively.makeOtherConstraints(currentFieldIndex);

        // Check if Dirichlet BCs are used
        for (unsigned int i = 0; i < userInputs.BC_list.size(); i++) {
            for (unsigned int direction = 0; direction < 2 * dim; direction++) {
                if (userInputs.BC_list[i].var_BC_type[direction] == DIRICHLET) {
                    it->hasDirichletBCs = true;
                } else if (userInputs.BC_list[i].var_BC_type[direction] == NON_UNIFORM_DIRICHLET) {
                    it->hasnonuniformDirichletBCs = true;
                } else if (userInputs.BC_list[i].var_BC_type[direction] == NEUMANN) {
                    it->hasNeumannBCs = true;
                }
            }
        }
        
        // Get constraints for Dirichlet BCs
        applyDirichletBCs();

        // Store Dirichlet BC DOF's
        BCs.valuesDirichletSet[it->index]->clear();
        for (types::global_dof_index i = 0; i < Discretization.dofHandlersSet[currentFieldIndex]->n_dofs(); i++) {
            if (Discretization.locally_relevant_dofsSet[currentFieldIndex]->is_element(i)) {
                if (BCs.constraintsDirichletSet[currentFieldIndex]->is_constrained(i)) {
                    (*BCs.valuesDirichletSet[it->index])[i] = BCs.constraintsDirichletSet[currentFieldIndex]->get_inhomogeneity(i);
                }
            }
        }

        snprintf(buffer, sizeof(buffer), "field '%2s' DOF : %u (Constraint DOF : %u)\n",
            it->name.c_str(), Discretization.dofHandlersSet[currentFieldIndex]->n_dofs(), BCs.constraintsDirichletSet[currentFieldIndex]->n_constraints());
        pcout << buffer;
    }
    pcout << "total DOF : " << Discretization.totalDOFs << std::endl;

    // Setup the matrix free object
    typename MatrixFree<dim, double>::AdditionalData additional_data;
// The member "mpi_communicator" was removed in deal.II version 8.5 but is required before it
#if (DEAL_II_VERSION_MAJOR < 9 && DEAL_II_VERSION_MINOR < 5)
    additional_data.mpi_communicator = MPI_COMM_WORLD;
#endif
    additional_data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::partition_partition;
    // additional_data.tasks_parallel_scheme = MatrixFree<dim,double>::AdditionalData::none;
    additional_data.mapping_update_flags = (update_values | update_gradients | update_JxW_values | update_quadrature_points);
    QGaussLobatto<1> quadrature(degree + 1);
    Discretization.matrixFreeObject.clear();
#if (DEAL_II_VERSION_MAJOR == 9 && DEAL_II_VERSION_MINOR < 4)
    Discretization.matrixFreeObject.reinit(Discretization.dofHandlersSet, RefineAdaptively.constraintsOtherSet, quadrature, additional_data);
#else
    Discretization.matrixFreeObject.reinit(MappingFE<dim, dim>(FE_Q<dim>(QGaussLobatto<1>(degree + 1))),
        Discretization.dofHandlersSet, RefineAdaptively.constraintsOtherSet, quadrature, additional_data);
#endif
    bool dU_scalar_init = false;
    bool dU_vector_init = false;

    // Setup solution vectors
    pcout << "initializing parallel::distributed residual and solution vectors\n";
    for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++) {
        vectorType *U, *R;

        U = new vectorType;
        R = new vectorType;
        tStep.solutionSet.push_back(U);
        residualSet.push_back(R);
        Discretization.matrixFreeObject.initialize_dof_vector(*R, fieldIndex);
        *R = 0;

        Discretization.matrixFreeObject.initialize_dof_vector(*U, fieldIndex);
        *U = 0;

        // Initializing temporary dU vector required for implicit solves of the elliptic equation.
        if (fields[fieldIndex].pdetype == TIME_INDEPENDENT || fields[fieldIndex].pdetype == IMPLICIT_TIME_DEPENDENT || (fields[fieldIndex].pdetype == AUXILIARY && userInputs.var_nonlinear[fieldIndex])) {
            if (fields[fieldIndex].type == SCALAR) {
                if (dU_scalar_init == false) {
                    Discretization.matrixFreeObject.initialize_dof_vector(dU_scalar, fieldIndex);
                    dU_scalar_init = true;
                }
            } else {
                if (dU_vector_init == false) {
                    Discretization.matrixFreeObject.initialize_dof_vector(dU_vector, fieldIndex);
                    dU_vector_init = true;
                }
            }
        }
    }

    // check if time dependent BVP and compute invM
    if (pFlags.isTimeDependentBVP) {
        computeInvM();
    }

    // Apply the initial conditions to the solution vectors
    // The initial conditions are re-applied below in the "adaptiveRefine" function so that the mesh can
    // adapt based on the initial conditions.
    if (userInputs.resume_from_checkpoint) {
        checkpoints.load_checkpoint_fields();
    } else {
        applyInitialConditions();
    }

    // Create new solution transfer sets (needed for the "refineGrid" call, might be able to move this elsewhere)
    soltransSet.clear();
    for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++) {
        soltransSet.push_back(new parallel::distributed::SolutionTransfer<dim, vectorType>(*Discretization.dofHandlersSet_nonconst[fieldIndex]));
    }

    // Ghost the solution vectors. Also apply the constraints (if any) on the solution vectors
    for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++) {
        BCs.constraintsDirichletSet[fieldIndex]->distribute(*tStep.solutionSet[fieldIndex]);
        RefineAdaptively.constraintsOtherSet[fieldIndex]->distribute(*tStep.solutionSet[fieldIndex]);
        tStep.solutionSet[fieldIndex]->update_ghost_values();
    }

    // If not resuming from a checkpoint, check and perform adaptive mesh refinement, which reinitializes the system with the new mesh
    if (!userInputs.resume_from_checkpoint && userInputs.h_adaptivity == true) {
        computing_timer.enter_subsection("matrixFreePDE: AMR");

        unsigned int numDoF_preremesh = Discretization.totalDOFs;
        for (unsigned int remesh_index = 0; remesh_index < (userInputs.max_refinement_level - userInputs.min_refinement_level); remesh_index++) {
            RefineAdaptively.adaptiveRefine(tStep.currentIncrement);
            reinit();
            if (Discretization.totalDOFs == numDoF_preremesh)
                break;
            numDoF_preremesh = Discretization.totalDOFs;
        }

        computing_timer.leave_subsection("matrixFreePDE: AMR");
    }

    // If resuming from a checkpoint, load the proper starting increment and time
    if (userInputs.resume_from_checkpoint) {
        checkpoints.load_checkpoint_time_info();
    }

    computing_timer.leave_subsection("matrixFreePDE: initialization");
}

#include "../../include/matrixFreePDE_template_instantiations.h"
