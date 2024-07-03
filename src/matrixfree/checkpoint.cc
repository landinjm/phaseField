#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "../../include/matrixFreePDE.h"

#ifdef DEAL_II_WITH_ZLIB
#include <zlib.h>
#endif



// Load from a previously created checkpoint
template <int dim, int degree>
void MatrixFreePDE<dim, degree>::load_checkpoint_triangulation()
{

    // First check existence of the two restart files for the mesh and field variables
    verify_checkpoint_file_exists("restart.mesh");
    verify_checkpoint_file_exists("restart.mesh.info");

    pcout << std::endl
          << "*** Resuming from a checkpoint! ***" << std::endl
          << std::endl;

    try {
        Discretization.triangulation.load("restart.mesh");
    } catch (...) {
        AssertThrow(false, ExcMessage("PRISMS-PF Error: Cannot open snapshot mesh file or read the triangulation stored there."));
    }
}

// Load from a previously saved checkpoint
template <int dim, int degree>
void MatrixFreePDE<dim, degree>::load_checkpoint_fields()
{

    // Serializing all of the scalars together and all of the vectors together

    // First, get lists of scalar and vector fields
    std::vector<unsigned int> scalar_var_indices, vector_var_indices;
    for (unsigned int var = 0; var < userInputs.number_of_variables; var++) {
        if (userInputs.var_type[var] == SCALAR) {
            scalar_var_indices.push_back(var);
        } else {
            vector_var_indices.push_back(var);
        }
    }

    // Second, build one solution set list for scalars and one for vectors
    std::vector<vectorType*> solSet_transfer_scalars;
    std::vector<vectorType*> solSet_transfer_vectors;
    for (unsigned int var = 0; var < userInputs.number_of_variables; ++var) {
        if (userInputs.var_type[var] == SCALAR) {
            solSet_transfer_scalars.push_back(tStep.solutionSet[var]);
        } else {
            solSet_transfer_vectors.push_back(tStep.solutionSet[var]);
        }
    }

    // Finally, deserialize the fields to the solSet_transfer objects, which contain pointers to solutionSet
    if (scalar_var_indices.size() > 0) {
        parallel::distributed::SolutionTransfer<dim, vectorType> system_trans_scalars(*Discretization.dofHandlersSet[scalar_var_indices[0]]);
        system_trans_scalars.deserialize(solSet_transfer_scalars);
    }
    if (vector_var_indices.size() > 0) {
        parallel::distributed::SolutionTransfer<dim, vectorType> system_trans_vectors(*Discretization.dofHandlersSet[vector_var_indices[0]]);
        system_trans_vectors.deserialize(solSet_transfer_vectors);
    }
}

// Load from a previously saved checkpoint
template <int dim, int degree>
void MatrixFreePDE<dim, degree>::load_checkpoint_time_info()
{

    // Make sure that restart.time.info exists
    verify_checkpoint_file_exists("restart.time.info");

    std::ifstream time_info_file;
    time_info_file.open("restart.time.info");
    std::string line;
    std::getline(time_info_file, line);
    line.erase(line.end() - 19, line.end());
    tStep.currentIncrement = dealii::Utilities::string_to_int(line);

    std::getline(time_info_file, line);
    line.erase(line.end() - 14, line.end());
    tStep.currentTime = dealii::Utilities::string_to_double(line);
    time_info_file.close();
}

template <int dim, int degree>
void MatrixFreePDE<dim, degree>::verify_checkpoint_file_exists(const std::string filename)
{
    std::ifstream in(filename);
    if (!in) {
        AssertThrow(false,
            ExcMessage(std::string("PRISMS-PF Error: You are trying to restart a previous computation, "
                                   "but the restart file <")
                + filename
                + "> does not appear to exist!"));
    }
}

#include "../../include/matrixFreePDE_template_instantiations.h"
