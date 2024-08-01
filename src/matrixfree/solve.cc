// solve() method for MatrixFreePDE class

#include "../../include/matrixFreePDE.h"

// solve BVP
template <int dim, int degree>
void MatrixFreePDE<dim, degree>::solve()
{
    // log time
    computing_timer.enter_subsection("matrixFreePDE: solve");
    pcout << "\nsolving...\n\n";

    // time dependent BVP
    if (isTimeDependentBVP) {

        // If grain reassignment is activated, reassign grains
        if (userInputs.grain_remapping_activated and (tStep.currentIncrement % userInputs.skip_grain_reassignment_steps == 0 or tStep.currentIncrement == 0)) {
            reassignGrains();
        }

        // For any nonlinear equation, set the initial guess as the solution to Laplace's equations
        generatingInitialGuess = true;
        setNonlinearEqInitialGuess();
        generatingInitialGuess = false;

        // Do an initial solve to set the elliptic fields
        solveIncrement(true);

        // output initial conditions for time dependent BVP
        if (userInputs.outputTimeStepList[currentOutput] == tStep.currentIncrement) {

            for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++) {
                BCs.constraintsDirichletSet[fieldIndex]->distribute(*tStep.solutionSet[fieldIndex]);
                RefineAdaptively.constraintsOtherSet[fieldIndex]->distribute(*tStep.solutionSet[fieldIndex]);
                tStep.solutionSet[fieldIndex]->update_ghost_values();
            }
            outputResults();
            currentOutput++;
        }

        if (userInputs.checkpointTimeStepList[checkpoints.currentCheckpoint] == tStep.currentIncrement) {
            computing_timer.enter_subsection("matrixFreePDE: save_checkpoint");
            checkpoints.save_checkpoint();
            computing_timer.leave_subsection("matrixFreePDE: save_checkpoint");
            checkpoints.currentCheckpoint++;
        }

        // Increase the current increment from 0 to 1 now that the initial conditions have been output
        tStep.currentIncrement++;

        // Cycle up to the proper output and checkpoint counters
        while (userInputs.outputTimeStepList.size() > 0 && userInputs.outputTimeStepList[currentOutput] < tStep.currentIncrement) {
            currentOutput++;
        }
        while (userInputs.checkpointTimeStepList.size() > 0 && userInputs.checkpointTimeStepList[checkpoints.currentCheckpoint] < tStep.currentIncrement) {
            checkpoints.currentCheckpoint++;
        }

        // time stepping
        pcout << "\nTime stepping parameters: timeStep: " << userInputs.dtValue << "  timeFinal: " << userInputs.finalTime << "  timeIncrements: " << userInputs.totalIncrements << "\n";

        // This is the main time-stepping loop
        for (; tStep.currentIncrement <= userInputs.totalIncrements; ++tStep.currentIncrement) {
            // increment current time
            tStep.currentTime += userInputs.dtValue;
            if (tStep.currentIncrement % userInputs.skip_print_steps == 0) {
                pcout << "\ntime increment:" << tStep.currentIncrement << "  time: " << tStep.currentTime << "\n";
            }

            // check and perform adaptive mesh refinement
            if (userInputs.h_adaptivity == true && tStep.currentIncrement % userInputs.skip_remeshing_steps == 0) {
                computing_timer.enter_subsection("matrixFreePDE: AMR");

                RefineAdaptively.adaptiveRefine(tStep.currentIncrement);
                reinit();

                computing_timer.leave_subsection("matrixFreePDE: AMR");
            }

            // Update the list of nuclei (if relevant)
            updateNucleiList();

            // If grain reassignment is activated, reassign grains
            if (userInputs.grain_remapping_activated and (tStep.currentIncrement % userInputs.skip_grain_reassignment_steps == 0 or tStep.currentIncrement == 0)) {
                reassignGrains();
            }

            // solve time increment
            solveIncrement(false);

            // Output results to file (on the proper increments)
            if (userInputs.outputTimeStepList[currentOutput] == tStep.currentIncrement) {
                for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++) {
                    BCs.constraintsDirichletSet[fieldIndex]->distribute(*tStep.solutionSet[fieldIndex]);
                    RefineAdaptively.constraintsOtherSet[fieldIndex]->distribute(*tStep.solutionSet[fieldIndex]);
                    tStep.solutionSet[fieldIndex]->update_ghost_values();
                }
                outputResults();
                if (userInputs.print_timing_with_output && tStep.currentIncrement < userInputs.totalIncrements) {
                    computing_timer.print_summary();
                }

                currentOutput++;
            }

            // Create a checkpoint (on the proper increments)
            if (userInputs.checkpointTimeStepList[checkpoints.currentCheckpoint] == tStep.currentIncrement) {
                computing_timer.enter_subsection("matrixFreePDE: save_checkpoint");
                checkpoints.save_checkpoint();
                computing_timer.leave_subsection("matrixFreePDE: save_checkpoint");
                checkpoints.currentCheckpoint++;
            }
        }
    }

    // time independent BVP
    else {
        generatingInitialGuess = false;

        // solve
        solveIncrement(false);

        // output results to file
        outputResults();
    }

    // log time
    computing_timer.leave_subsection("matrixFreePDE: solve");
}

#include "../../include/matrixFreePDE_template_instantiations.h"
