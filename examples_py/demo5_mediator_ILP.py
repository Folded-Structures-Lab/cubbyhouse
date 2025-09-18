

import pandas as pd
from cubbyhouse.mediator import CubbyhouseMediator, DispatchStage
from cubbyhouse.utils import get_stock_file_from_case_id, get_structure_from_case_id, IO_FOLDERS
from cubbyhouse.boardstock import BoardStock
from cubbyhouse.opt import solve_ilp
pd.set_option('future.no_silent_downcasting', True) #related to fillna future depreciation error

PRINT_ME = True
EXPORT_ME = False
CASE = "case_1"


####################################
# CREATE STRUCTURE, STOCK, MEDIATOR
####################################
structure = get_structure_from_case_id(CASE)
board_stock = BoardStock.from_element_group_csv(get_stock_file_from_case_id(CASE))
mediator = CubbyhouseMediator(structure, board_stock)

################################
# SET-BASED DISPATCH: STAGE 3
################################
print("\nSet-based dispatch Stage 3:")
if CASE=="case_2":
    mediator.THRESHOLD_SCORE = 7.8
    mediator.MEMBER_MULTIPLIER=6
mediator.dispatch_section_groups(DispatchStage.STAGE3_FINAL_ALLOCATION)

################################
# PATTERN AND MATRIX GENERATION
################################
print("\n Member Jointing and Element Cutting Pattern Generation:")
mediator.create_members() #create members including member jointing patterns
mediator.create_element_group_cutting_patterns(pattern_type = 'no_usable_residual')

#############
# ASSIGNMENT
#############
objective, result_x, result_y, constraints = solve_ilp(mediator)


if PRINT_ME:

    print("\n Jointing Pattern Solution:")
    print(mediator.compile_jointing_results(result_x))
    #Cutting Pattern Summary
    print("\n Parts produced per member group:")
    print(mediator.q_pm(result_x))
    print("\n Member results summary:")
    print(mediator.compile_member_results(result_x))

    print("\n Cutting Pattern Solution:")
    print(mediator.compile_cutting_results(result_y))
    print("\n Parts produced per element group:")
    print(mediator.q_pi(result_y))

    # print("\n Section group results summary:")
    # print(mediator.compile_section_group_results(result_y))


    print("\n Element group results summary:")
    print(mediator.compile_element_group_results(result_y))


    print("\n Unused parts per element group:")
    print(mediator.compile_part_results(result_x,result_y))

    print("\n Waste produced per element group:")
    print(mediator.w_i(result_x,result_y))




if EXPORT_ME:
    results_dir = f"{IO_FOLDERS.RESULTS.value}{CASE}"

    #raw_cutting_files

    mediator.export_cutting_results(result_y, results_dir)
    mediator.export_cutting_results_raw(result_y, results_dir)
    # test = mediator.export_cutting_results( result_y)
    mediator.export_jointing_results( result_x, results_dir)
    mediator.export_jointing_results_raw( result_x, results_dir)
    # test = mediator.export_jointing_results( result_x)

    mediator.export_member_results_summary(result_x, results_dir)
    # mediator.export_section_group_results_summary(result_y, results_dir)
    mediator.export_element_group_results_summary(result_y, results_dir)

    mediator.export_part_results(result_x,result_y, results_dir)
    mediator.export_wastage_results(result_x,result_y, results_dir)



