
from pulp import LpMinimize, LpProblem, LpVariable, lpSum
from cubbyhouse.mediator import CubbyhouseMediator


def solve_ilp(mediator: CubbyhouseMediator):
    jointing_patterns_df = mediator.J_matrix
    cutting_patterns_df = mediator.C_matrix
    element_demand_df = mediator.E_matrix 
    element_quantities_df = mediator.Q_e
    member_supply_df = mediator.M_matrix
    member_quantities_df = mediator.Q_m
    waste_df = mediator.w_vector
    part_length_df = mediator.l_vector
    member_multiplier = mediator.MEMBER_MULTIPLIER
    member_quantities_df = member_quantities_df *member_multiplier
    
    # Create the model
    model = LpProblem(name="assignment-problem", sense=LpMinimize)

    # Initialize the decision variables for member jointing patterns
    x = {j: LpVariable(name=f"x_{j}", lowBound=0, cat="Integer") for j in jointing_patterns_df.index}
    
    # Initialize the decision variables for element cutting patterns
    y = {c: LpVariable(name=f"x_c_{c}", lowBound=0, cat="Integer") for c in cutting_patterns_df.index}

    # Add the objective function to the model

    # model += (lpSum(y[c] * waste_df.at[c, 'waste'] for c in cutting_patterns_df.index), "Objective Function")

    # Add the objective function to the model
    model += (
        lpSum(y[c] * waste_df.at[c, 'waste'] for c in cutting_patterns_df.index) +
        lpSum(
            (lpSum(cutting_patterns_df.at[c, part_id] * y[c] for c in cutting_patterns_df.index) - 
             lpSum(jointing_patterns_df.at[j, part_id] * x[j] for j in jointing_patterns_df.index)) *
            part_length_df.at[part_id, 'part_length']
            for part_id in jointing_patterns_df.columns
        ) + 
        #minimise number of parts
        lpSum(
            lpSum(jointing_patterns_df.at[j, part_id] * x[j] for j in jointing_patterns_df.index) 
            for part_id in jointing_patterns_df.columns
        ),
        "Objective Function"
    )

    # Add constraints to ensure part production meets or exceeds part requirements
    for part_id in jointing_patterns_df.columns:
        model += (
            lpSum(jointing_patterns_df.at[j, part_id] * x[j] for j in jointing_patterns_df.index) <= 
            lpSum(cutting_patterns_df.at[c, part_id] * y[c] for c in cutting_patterns_df.index),
            f"Part_Requirement_{part_id}"
        )

    # Add constraints to ensure element usage does not exceed available quantities
    for element_id in element_quantities_df.index:
        model += (
            lpSum(element_demand_df.at[c, element_id] * y[c] for c in element_demand_df.index) <= element_quantities_df.at[element_id, 'qty'],
            f"Element_Demand_{element_id}"
        )

    # Add constraints to ensure members produce meet required demand
    for member_id in member_quantities_df.index:
        model += (
            lpSum(member_supply_df.at[j, member_id] * x[j] for j in member_supply_df.index) == member_quantities_df.at[member_id, 'qty'],
            f"Member_Supply_{member_id}"
        )


    # Solve the optimization problem
    model.solve()

    # Get the results
    result_x = {f"x_{j}": x[j].value() for j in jointing_patterns_df.index}
    result_y = {f"y_{c}": y[c].value() for c in cutting_patterns_df.index}

    # Print the results
    # print("Optimal values for member jointing patterns (x_i):")
    # for var, value in result_x.items():
    #     print(f"{var}: {value}")

    # print("\nOptimal values for element cutting patterns (y_j):")
    # for var, value in result_y.items():
    #     print(f"{var}: {value}")

    print(f"\nMaximum value of the objective function: {model.objective.value()}")

    # Print the constraint values
    # print("\nConstraint values:")
    # for name, constraint in model.constraints.items():
    #     print(f"{name}: {constraint.value()}")

    # print("\n Members produced:")

    return model.objective.value(), result_x, result_y, model.constraints
    #print(sum([x[i])
