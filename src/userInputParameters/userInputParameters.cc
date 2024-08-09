// Methods for the userInputParameters class
#include "../../include/userInputParameters.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/exceptions.h>

using namespace dealii;

template <int dim>
userInputParameters<dim>::userInputParameters(inputFileReader& input_file_reader, dealii::ParameterHandler& parameter_handler, variableAttributeLoader variable_attributes)
{

    // Load the inputs into the class member variables
    loadVariableAttributes(variable_attributes);

    /*
        Meshing Parameters
    */

    // Coordinate Axes
    std::vector<std::string> coordAxes = {"X", "Y", "Z"};

    // Domain size
    for (unsigned int i = 0; i < dim; ++i) {
        domain_size.push_back(parameter_handler.get_double("Domain size " + coordAxes[i]));
    }

    // Subdivisions
    for (unsigned int i = 0; i < dim; ++i) {
        subdivisions.push_back(parameter_handler.get_integer("Subdivisions " + coordAxes[i]));
    }

    // Global refinement
    refine_factor = parameter_handler.get_integer("Refine factor");

    // Element degree
    degree = parameter_handler.get_integer("Element degree");

    // Flag for AMR 
    h_adaptivity = parameter_handler.get_bool("Mesh adaptivity");

    // Steps between remeshing
    skip_remeshing_steps = parameter_handler.get_integer("Steps between remeshing operations");

    // Minimum AMR refinement level
    max_refinement_level = parameter_handler.get_integer("Max refinement level");

    // Maximum AMR refinement level
    min_refinement_level = parameter_handler.get_integer("Min refinement level");

    // Check that the initial global refinement level is between the max and min AMR levels
    if (h_adaptivity && ((refine_factor < min_refinement_level) || (refine_factor > max_refinement_level))) {
        AssertThrow(false, ExcMessage("The <Refine factor> parameter has an invalid value. It must be between the minimum and maximum refinement levels when AMR is enabled"));
    }

    // The adaptivity criterion for each variable has its own subsection
    for (unsigned int i = 0; i < number_of_variables; i++) {

        // Enter the refinement subsection for variable i, even if it does not exist 
        std::string subsection_text = "Refinement criterion: ";
        subsection_text.append(input_file_reader.var_names.at(i));

        parameter_handler.enter_subsection(subsection_text);
        
        std::string crit_type_string = parameter_handler.get("Criterion type");

        // If no criterion exists skips the following steps and continue in the loop
        if (!(crit_type_string.size() > 0)) {
            parameter_handler.leave_subsection();
            continue;
        }

        // Create and fill an instance of the refinement criterion for this variable 
        RefinementCriterion new_criterion;
        new_criterion.variable_index = i;
        new_criterion.variable_name = input_file_reader.var_names.at(i);

        if (boost::iequals(crit_type_string, "VALUE")) {
            new_criterion.criterion_type = VALUE;
            new_criterion.value_lower_bound = parameter_handler.get_double("Value lower bound");
            new_criterion.value_upper_bound = parameter_handler.get_double("Value upper bound");
        } else if (boost::iequals(crit_type_string, "GRADIENT")) {
            new_criterion.criterion_type = GRADIENT;
            new_criterion.gradient_lower_bound = parameter_handler.get_double("Gradient magnitude lower bound");
        } else if (boost::iequals(crit_type_string, "VALUE_AND_GRADIENT")) {
            new_criterion.criterion_type = VALUE_AND_GRADIENT;
            new_criterion.value_lower_bound = parameter_handler.get_double("Value lower bound");
            new_criterion.value_upper_bound = parameter_handler.get_double("Value upper bound");
            new_criterion.gradient_lower_bound = parameter_handler.get_double("Gradient magnitude lower bound");
        } else {
            AssertThrow(false, ExcMessage("The <Criterion type> parameter has an invalid value. The allowed types are VALUE, GRADIENT, VALUE_AND_GRADIENT."));
        }

        // Check to make sure that the upper bound is greater than or equal to the lower bound
        bool containsValue = new_criterion.criterion_type == VALUE || new_criterion.criterion_type == VALUE_AND_GRADIENT;
        bool LowerBoundIsLowerBound = new_criterion.value_lower_bound < new_criterion.value_upper_bound;
        if (containsValue && !LowerBoundIsLowerBound) {
            AssertThrow(false, ExcMessage("The <Value upper bound> parameter has an invalid value. The lower bound must be lower than the upper bound."));
        }

        refinement_criteria.push_back(new_criterion);
        
        parameter_handler.leave_subsection();
    }

    /*
        Time Stepping Parameters
    */

    // Time step
    dtValue = parameter_handler.get_double("Time step");

    // Check that the time step is non-negative
    if (dtValue < 0.0) {
        AssertThrow(false, ExcMessage("The <Time step> parameter has an invalid value. The time step must be greater than or equal to 0."));
    }

    // Total number of increments
    unsigned int totalIncrements_temp = parameter_handler.get_integer("Number of time steps");

    // Final simulation time
    finalTime = parameter_handler.get_double("Simulation end time");

    // Check that the time step is non-negative
    if (finalTime < 0.0) {
        AssertThrow(false, ExcMessage("The <Simulation end time> parameter has an invalid value. The final simulation time must be greater than or equal to 0."));
    }

    // Determine the maximum number of time steps
    bool defaultFinalTime = finalTime == 0.0;
    bool defaultTimeStep = dtValue == 0.0;

    if (defaultFinalTime && defaultTimeStep) {
        totalIncrements = 0;
    } else if (defaultFinalTime) {
        totalIncrements = totalIncrements_temp;
        finalTime = totalIncrements * dtValue;
    } else if (defaultTimeStep) {
        totalIncrements = std::ceil(finalTime / dtValue);
    } else {
        if (std::ceil(finalTime / dtValue) < totalIncrements_temp) {
            totalIncrements = totalIncrements_temp;
            finalTime = totalIncrements * dtValue;
        } else {
            totalIncrements = std::ceil(finalTime / dtValue);
        }
    }

    /*
        Linear Solver Parameters
    */

    for (unsigned int i = 0; i < number_of_variables; i++) {

        // If variable doesn't require a linear solve continue in the loop
        // This could be moved down so the user is aware if they incorrectly specify the linear solver parameters
        // (i.e., setting the linear solve for a explicit field)
        if (!(input_file_reader.var_eq_types.at(i) == IMPLICIT_TIME_DEPENDENT) && !(input_file_reader.var_eq_types.at(i) == TIME_INDEPENDENT)) {
            continue;
        }

        // Enter the linear solver subsection for variable i 
        std::string subsection_text = "Linear solver parameters: ";
        subsection_text.append(input_file_reader.var_names.at(i));

        parameter_handler.enter_subsection(subsection_text);
        
        // Tolerance type
        SolverToleranceType temp_type;
        std::string type_string = parameter_handler.get("Tolerance type");

        if (boost::iequals(type_string, "ABSOLUTE_RESIDUAL")) {
            temp_type = ABSOLUTE_RESIDUAL;
        } else if (boost::iequals(type_string, "RELATIVE_RESIDUAL_CHANGE")) {
            temp_type = RELATIVE_RESIDUAL_CHANGE;
        } else if (boost::iequals(type_string, "ABSOLUTE_SOLUTION_CHANGE")) {
            temp_type = ABSOLUTE_SOLUTION_CHANGE;
            AssertThrow(false, ExcMessage("The <Tolerance type> parameter has an invalid value. The absolute solution change is not currently supported."));
        } else {
            AssertThrow(false, ExcMessage("The <Tolerance type> parameter has an invalid value. Only ABSOLUTE_RESIDUAL, RELATIVE_RESIDUAL_CHANGE, and ABSOLUTE_SOLUTION_CHANGE are allowed."));
        }

        // Tolerance value
        double temp_value = parameter_handler.get_double("Tolerance value");

        // Maximum number of iterations
        unsigned int temp_max_iterations = parameter_handler.get_integer("Maximum linear solver iterations");

        // Load the parameters and leave the subsection
        linear_solver_parameters.loadParameters(i, temp_type, temp_value, temp_max_iterations);
        
        parameter_handler.leave_subsection();
    }

    /*
        Non-linear Solver Parameters
    */

    std::vector<bool> var_nonlinear = variable_attributes.var_nonlinear;

    nonlinear_solver_parameters.setMaxIterations(parameter_handler.get_integer("Maximum nonlinear solver iterations"));

    for (unsigned int i = 0; i < var_nonlinear.size(); i++) {

        // If variable doesn't require a nonlinear solve continue in the loop
        if (!var_nonlinear.at(i)) {
            continue;
        }

        // Enter the linear solver subsection for variable i 
        std::string subsection_text = "Nonlinear solver parameters: ";
        subsection_text.append(input_file_reader.var_names.at(i));

        parameter_handler.enter_subsection(subsection_text);

        // Set the tolerance type
        SolverToleranceType temp_type;
        std::string type_string = parameter_handler.get("Tolerance type");

        if (boost::iequals(type_string, "ABSOLUTE_RESIDUAL")) {
            temp_type = ABSOLUTE_RESIDUAL;
        } else if (boost::iequals(type_string, "RELATIVE_RESIDUAL_CHANGE")) {
            temp_type = RELATIVE_RESIDUAL_CHANGE;
        } else if (boost::iequals(type_string, "ABSOLUTE_SOLUTION_CHANGE")) {
            temp_type = ABSOLUTE_SOLUTION_CHANGE;
        } else {
            AssertThrow(false, ExcMessage("The <Tolerance type> parameter has an invalid value. Only ABSOLUTE_RESIDUAL, RELATIVE_RESIDUAL_CHANGE, and ABSOLUTE_SOLUTION_CHANGE are allowed."));
        }

        // Tolerance value
        double temp_value = parameter_handler.get_double("Tolerance value");

        // Backtrace damping flag
        bool temp_backtrack_damping = parameter_handler.get_bool("Use backtracking line search damping");

        // Backtracking step size modifier
        double temp_step_modifier = parameter_handler.get_double("Backtracking step size modifier");

        // Constant that determines how much the residual must decrease to be accepted as sufficient
        double temp_residual_decrease_coeff = parameter_handler.get_double("Backtracking residual decrease coefficient");

        // Default damping coefficient (used if backtracking isn't used)
        double temp_damping_coefficient = parameter_handler.get_double("Constant damping value");

        // Whether to use the solution of Laplace's equation instead of the IC in ICs_and_BCs.h as the initial guess for nonlinear, time independent equations
        bool temp_laplace_for_initial_guess = false;
        if (var_eq_type[i] == TIME_INDEPENDENT) {
            temp_laplace_for_initial_guess = parameter_handler.get_bool("Use Laplace's equation to determine the initial guess");
        } else {
            if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
                std::cout << "PRISMS-PF Warning: Laplace's equation is only used to generate the initial guess for time independent equations. The equation for variable " << var_name[i] << " is not a time independent equation. No initial guess is needed for this equation." << std::endl;
            }
        }

        // Load the parameters and leave the subsection
        nonlinear_solver_parameters.loadParameters(i, temp_type, temp_value, temp_backtrack_damping, temp_step_modifier, temp_residual_decrease_coeff, temp_damping_coefficient, temp_laplace_for_initial_guess);

        parameter_handler.leave_subsection();
    }

    // Correction for the max number of nonlinear iterations if there are no nonlinear variables
    if (var_nonlinear.size() == 0) {
        nonlinear_solver_parameters.setMaxIterations(0);
    }

    // Output parameters
    std::string output_condition = parameter_handler.get("Output condition");
    unsigned int num_outputs = parameter_handler.get_integer("Number of outputs");
    std::vector<int> user_given_time_step_list_temp = dealii::Utilities::string_to_int(dealii::Utilities::split_string_list(parameter_handler.get("List of time steps to output")));
    std::vector<unsigned int> user_given_time_step_list;
    for (unsigned int i = 0; i < user_given_time_step_list_temp.size(); i++)
        user_given_time_step_list.push_back(user_given_time_step_list_temp[i]);

    skip_print_steps = parameter_handler.get_integer("Skip print steps");
    output_file_type = parameter_handler.get("Output file type");
    output_file_name = parameter_handler.get("Output file name (base)");

    output_vtu_per_process = parameter_handler.get_bool("Output separate files per process");
    if ((output_file_type == "vtk") && (!output_vtu_per_process)) {
        output_vtu_per_process = true;
        if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
            std::cout << "PRISMS-PF Warning: 'Output file type' given as 'vtk' and 'Output separate files per process' given as 'false'. Shared output files are not supported for the vtk output format. Separate files per process will be created." << std::endl;
        }
    }

    print_timing_with_output = parameter_handler.get_bool("Print timing information with output");

    const std::string OutputFileCompression = parameter_handler.get("Output file compression type");
    if (OutputFileCompression == "none") {
        filecompression = outputCompression::DEFAULT;
    } else if (OutputFileCompression == "default") {
        filecompression = outputCompression::SPEED;
    } else if (OutputFileCompression == "speed") {
        filecompression = outputCompression::SIZE;
    } else if (OutputFileCompression == "size") {
        filecompression = outputCompression::NONE;
    } else {
        AssertThrow(false, ExcMessage("The <Output file compression type> parameter has an invalid value."));
    }

    // Use these inputs to create a list of time steps where the code should output, stored in the member
    outputTimeStepList = setTimeStepList(output_condition, num_outputs, user_given_time_step_list);

    // Variables for loading in PField ICs
    std::vector<std::string> load_ICs_temp = dealii::Utilities::split_string_list(parameter_handler.get("Load initial conditions"));
    std::vector<std::string> load_parallel_file_temp = dealii::Utilities::split_string_list(parameter_handler.get("Load parallel file"));

    if (boost::iequals(load_ICs_temp.at(0), "void")) {
        for (unsigned int var = 0; var < number_of_variables; var++) {
            load_ICs.push_back(false);
            load_parallel_file.push_back(false);
        }
    } else {
        for (unsigned int var = 0; var < number_of_variables; var++) {
            if (boost::iequals(load_ICs_temp.at(var), "true")) {
                load_ICs.push_back(true);
            } else {
                load_ICs.push_back(false);
            }
            if (boost::iequals(load_parallel_file_temp.at(var), "true")) {
                load_parallel_file.push_back(true);
            } else {
                load_parallel_file.push_back(false);
            }
        }
    }

    load_file_name = dealii::Utilities::split_string_list(parameter_handler.get("File names"));
    load_field_name = dealii::Utilities::split_string_list(parameter_handler.get("Variable names in the files"));

    // Parameters for checkpoint/restart
    resume_from_checkpoint = parameter_handler.get_bool("Load from a checkpoint");
    std::string checkpoint_condition = parameter_handler.get("Checkpoint condition");
    unsigned int num_checkpoints = parameter_handler.get_integer("Number of checkpoints");

    std::vector<int> user_given_checkpoint_time_step_list_temp = dealii::Utilities::string_to_int(dealii::Utilities::split_string_list(parameter_handler.get("List of time steps to save checkpoints")));
    std::vector<unsigned int> user_given_checkpoint_time_step_list;
    for (unsigned int i = 0; i < user_given_checkpoint_time_step_list_temp.size(); i++)
        user_given_checkpoint_time_step_list.push_back(user_given_checkpoint_time_step_list_temp[i]);

    checkpointTimeStepList = setTimeStepList(checkpoint_condition, num_checkpoints, user_given_checkpoint_time_step_list);

    // Parameters for nucleation

    for (unsigned int i = 0; i < input_file_reader.var_types.size(); i++) {
        if (input_file_reader.var_nucleates.at(i)) {
            std::string nucleation_text = "Nucleation parameters: ";
            nucleation_text.append(input_file_reader.var_names.at(i));

            parameter_handler.enter_subsection(nucleation_text);
            {
                unsigned int var_index = i;
                std::vector<double> semiaxes = dealii::Utilities::string_to_double(dealii::Utilities::split_string_list(parameter_handler.get("Nucleus semiaxes (x, y, z)")));
                std::vector<double> ellipsoid_rotation = dealii::Utilities::string_to_double(dealii::Utilities::split_string_list(parameter_handler.get("Nucleus rotation in degrees (x, y, z)")));
                std::vector<double> freeze_semiaxes = dealii::Utilities::string_to_double(dealii::Utilities::split_string_list(parameter_handler.get("Freeze zone semiaxes (x, y, z)")));
                double hold_time = parameter_handler.get_double("Freeze time following nucleation");
                double no_nucleation_border_thickness = parameter_handler.get_double("Nucleation-free border thickness");

                nucleationParameters<dim> temp(var_index, semiaxes, freeze_semiaxes, ellipsoid_rotation, hold_time, no_nucleation_border_thickness);
                nucleation_parameters_list.push_back(temp);

                // Validate nucleation input
                if (semiaxes.size() < dim || semiaxes.size() > 3) {
                    std::cerr << "PRISMS-PF Error: The number of nucleus semiaxes given in the 'parameters.in' file must be at least the number of dimensions and no more than 3." << std::endl;
                    abort();
                }
                if (freeze_semiaxes.size() < dim || freeze_semiaxes.size() > 3) {
                    std::cerr << "PRISMS-PF Error: The number of nucleation freeze zone semiaxes given in the 'parameters.in' file must be at least the number of dimensions and no more than 3." << std::endl;
                    abort();
                }
                if (ellipsoid_rotation.size() != 3) {
                    std::cerr << "PRISMS-PF Error: Exactly three nucleus rotation angles must be given in the 'parameters.in' file." << std::endl;
                    abort();
                }
            }
            parameter_handler.leave_subsection();
        }
    }
    for (unsigned int i = 0; i < nucleation_parameters_list.size(); i++) {
        nucleation_parameters_list_index[nucleation_parameters_list.at(i).var_index] = i;
    }

    if (parameter_handler.get("Minimum allowed distance between nuclei") != "-1") {
        min_distance_between_nuclei = parameter_handler.get_double("Minimum allowed distance between nuclei");
    } else if (nucleation_parameters_list.size() > 1) {
        min_distance_between_nuclei = 2.0 * (*(max_element(nucleation_parameters_list[0].semiaxes.begin(), nucleation_parameters_list[0].semiaxes.end())));
    }
    evolution_before_nucleation = parameter_handler.get_bool("Enable evolution before nucleation");
    // Implement multiple order parameter nucleation later
    // multiple_nuclei_per_order_parameter = parameter_handler.get_bool("Allow multiple nuclei per order parameter");
    nucleation_order_parameter_cutoff = parameter_handler.get_double("Order parameter cutoff value");
    steps_between_nucleation_attempts = parameter_handler.get_integer("Time steps between nucleation attempts");
    nucleation_start_time = parameter_handler.get_double("Nucleation start time");
    nucleation_end_time = parameter_handler.get_double("Nucleation end time");

    // Load the grain remapping parameters
    grain_remapping_activated = parameter_handler.get_bool("Activate grain reassignment");

    skip_grain_reassignment_steps = parameter_handler.get_integer("Time steps between grain reassignments");

    order_parameter_threshold = parameter_handler.get_double("Order parameter cutoff for grain identification");

    buffer_between_grains = parameter_handler.get_double("Buffer between grains before reassignment");
    if (buffer_between_grains < 0.0 && grain_remapping_activated == true) {
        std::cerr << "PRISMS-PF Error: If grain reassignment is activated, a non-negative buffer distance must be given. See the 'Buffer between grains before reassignment' entry in parameters.in." << std::endl;
        abort();
    }

    std::vector<std::string> variables_for_remapping_str = dealii::Utilities::split_string_list(parameter_handler.get("Order parameter fields for grain reassignment"));
    for (unsigned int field = 0; field < variables_for_remapping_str.size(); field++) {
        bool field_found = false;
        for (unsigned int i = 0; i < number_of_variables; i++) {
            if (boost::iequals(variables_for_remapping_str[field], variable_attributes.var_name_list[i].second)) {
                variables_for_remapping.push_back(variable_attributes.var_name_list[i].first);
                field_found = true;
                break;
            }
        }
        if (field_found == false && grain_remapping_activated == true) {
            std::cerr << "PRISMS-PF Error: Entries in the list of order parameter fields used for grain reassignment must match the variable names in equations.h." << std::endl;
            std::cerr << variables_for_remapping_str[field] << std::endl;
            abort();
        }
    }

    load_grain_structure = parameter_handler.get_bool("Load grain structure");
    grain_structure_filename = parameter_handler.get("Grain structure filename");
    grain_structure_variable_name = parameter_handler.get("Grain structure variable name");
    num_grain_smoothing_cycles = parameter_handler.get_integer("Number of smoothing cycles after grain structure loading");
    min_radius_for_loading_grains = parameter_handler.get_double("Minimum radius for loaded grains");

    // Load the boundary condition variables into list of BCs (where each element of the vector is one component of one variable)
    std::vector<std::string> list_of_BCs;
    for (unsigned int i = 0; i < number_of_variables; i++) {
        if (var_type[i] == SCALAR) {
            std::string bc_text = "Boundary condition for variable ";
            bc_text.append(var_name.at(i));
            list_of_BCs.push_back(parameter_handler.get(bc_text));
        } else {
            std::string bc_text = "Boundary condition for variable ";
            bc_text.append(var_name.at(i));
            bc_text.append(", x component");
            list_of_BCs.push_back(parameter_handler.get(bc_text));

            bc_text = "Boundary condition for variable ";
            bc_text.append(var_name.at(i));
            bc_text.append(", y component");
            list_of_BCs.push_back(parameter_handler.get(bc_text));

            if (dim > 2) {
                bc_text = "Boundary condition for variable ";
                bc_text.append(var_name.at(i));
                bc_text.append(", z component");
                list_of_BCs.push_back(parameter_handler.get(bc_text));
            }
        }
    }

    // Load the BC information from the strings into a varBCs object
    load_BC_list(list_of_BCs);

    // Load the user-defined constants
    load_user_constants(input_file_reader, parameter_handler);
}

// Template instantiations
#include "../../include/userInputParameters_template_instantiations.h"
