#ifndef INCLUDE_CHECKPOINT_H_
#define INCLUDE_CHECKPOINT_H_

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "discretization.h"
#include "timeStepping.h"
#include "userInputParameters.h"

#ifdef DEAL_II_WITH_ZLIB
#include <zlib.h>
#endif

using namespace dealii;

/**
 * This class deals with the checkpoints. Add more comments later
 */
template <int dim, int degree>
class Checkpoint {
public:
    Checkpoint(const userInputParameters<dim>& _userInputs, discretization<dim>& Discretization, TimeStepping<dim, degree>& tStep);

    unsigned int currentCheckpoint;

    void save_checkpoint();

    void move_file(const std::string&, const std::string&);

private:
    /*User inputs*/
    userInputParameters<dim> userInputs;

    /*Discretiziation*/
    discretization<dim>& DiscretizationRef;

    /*Timestepping*/
    TimeStepping<dim, degree>& tStepRef;
};

template <int dim, int degree>
Checkpoint<dim, degree>::Checkpoint(const userInputParameters<dim>& _userInputs, discretization<dim>& Discretization, TimeStepping<dim, degree>& tStep)
    : userInputs(_userInputs)
    , DiscretizationRef(Discretization)
    , tStepRef(tStep)
    , currentCheckpoint(0)
{
}

// Save a checkpoint
template <int dim, int degree>
void Checkpoint<dim, degree>::save_checkpoint()
{
    unsigned int my_id = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    if (my_id == 0) {
        // if we have previously written a snapshot, then keep the last
        // snapshot in case this one fails to save. Note: static variables
        // will only be initialized once per model run.
        static bool previous_snapshot_exists = (userInputs.resume_from_checkpoint == true);

        if (previous_snapshot_exists == true) {
            move_file("restart.mesh", "restart.mesh.old");
            move_file("restart.mesh.info", "restart.mesh.info.old");
            move_file("restart.time.info", "restart.time.info.old");
        }
        // from now on, we know that if we get into this
        // function again that a snapshot has previously
        // been written
        previous_snapshot_exists = true;
    }

    // save Triangulation and Solution vectors:
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
        std::vector<const vectorType*> solSet_transfer_scalars;
        std::vector<const vectorType*> solSet_transfer_vectors;
        for (unsigned int var = 0; var < userInputs.number_of_variables; ++var) {
            if (userInputs.var_type[var] == SCALAR) {
                solSet_transfer_scalars.push_back(tStepRef.solutionSet[var]);
            } else {
                solSet_transfer_vectors.push_back(tStepRef.solutionSet[var]);
            }
        }

        // Finally, save the triangulation and the solutionTransfer objects
        if (scalar_var_indices.size() > 0 && vector_var_indices.size() == 0) {
            parallel::distributed::SolutionTransfer<dim, vectorType> system_trans_scalars(*DiscretizationRef.dofHandlersSet[scalar_var_indices[0]]);
            system_trans_scalars.prepare_for_serialization(solSet_transfer_scalars);

            DiscretizationRef.triangulation.save("restart.mesh");
        } else if (scalar_var_indices.size() == 0 && vector_var_indices.size() > 0) {
            parallel::distributed::SolutionTransfer<dim, vectorType> system_trans_vectors(*DiscretizationRef.dofHandlersSet[vector_var_indices[0]]);
            system_trans_vectors.prepare_for_serialization(solSet_transfer_vectors);

            DiscretizationRef.triangulation.save("restart.mesh");
        } else {
            parallel::distributed::SolutionTransfer<dim, vectorType> system_trans_scalars(*DiscretizationRef.dofHandlersSet[scalar_var_indices[0]]);
            system_trans_scalars.prepare_for_serialization(solSet_transfer_scalars);

            parallel::distributed::SolutionTransfer<dim, vectorType> system_trans_vectors(*DiscretizationRef.dofHandlersSet[vector_var_indices[0]]);
            system_trans_vectors.prepare_for_serialization(solSet_transfer_vectors);

            DiscretizationRef.triangulation.save("restart.mesh");
        }
    }

    // Save information about the current increment and current time
    if (my_id == 0) {
        std::ofstream time_info_file;
        time_info_file.open("restart.time.info");
        time_info_file << tStepRef.currentIncrement << " (currentIncrement)\n";
        time_info_file << tStepRef.currentTime << " (currentTime)\n";
        time_info_file.close();
    }
}

// Move/rename a checkpoint file
template <int dim, int degree>
void Checkpoint<dim, degree>::move_file(const std::string& old_name, const std::string& new_name)
{

    int error = system(("mv " + old_name + " " + new_name).c_str());

    // If the above call failed, e.g. because there is no command-line
    // available, try with internal functions.
    if (error != 0) {
        std::ifstream ifile(new_name);
        if (static_cast<bool>(ifile)) {
            error = remove(new_name.c_str());
            AssertThrow(error == 0, ExcMessage(std::string("Unable to remove file: " + new_name + ", although it seems to exist. " + "The error code is " + dealii::Utilities::to_string(error) + ".")));
        }

        error = rename(old_name.c_str(), new_name.c_str());
        AssertThrow(error == 0, ExcMessage(std::string("Unable to rename files: ") + old_name + " -> " + new_name + ". The error code is " + dealii::Utilities::to_string(error) + "."));
    }
}

#endif
