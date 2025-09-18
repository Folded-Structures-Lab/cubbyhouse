

from cubbyhouse.mediator import CubbyhouseMediator, DispatchStage
from cubbyhouse.boardstock import BoardStock
from cubbyhouse.utils import get_stock_file_from_case_id, get_structure_from_case_id, IO_FOLDERS

PRINT_METHODS = True
EXPORT_ME = False
CASE = "case_1"

########################
# IMPORT STRUCTURE
########################
structure = get_structure_from_case_id(CASE)
print(f"Imported framing group data for {CASE} structure:")
print(structure.framing_group_df)
eg_framing_group_name = list(structure.framing_groups.keys())[0]
eg_framing_group = structure.framing_groups[eg_framing_group_name]



######################
# IMPORT BOARD STOCK
######################
case_stock = get_stock_file_from_case_id(CASE)
board_stock = BoardStock.from_element_group_csv(case_stock)
print(f"Imported section group data for {CASE} stock:")
section_group_data = board_stock.section_group_df
print(board_stock.section_group_df)

#############################
# CREATE CUBBYHOUSE MEDIATOR
#############################
mediator = CubbyhouseMediator(structure, board_stock)
if PRINT_METHODS:
    print('\n Mediator adds span table to each framing group:')
    print(eg_framing_group.span_table.head(10))

##############################
# SET-BASED DISPATCH: STAGE 1
# -> actions to reduce required length 
# -> reduce length & width extent, increase spacing, decrease tranverse lamination)
##############################
print("\nSet-based dispatch stage 0: all section_groups to all framing groups")
mediator.dispatch_section_groups(DispatchStage.STAGE0_ALL)
# dispatch = available length
print(mediator.dispatch_matrix)
# normalised_dispatch_matrix = available_length / req_member_length = (D_hat matrix)
print(mediator.normalised_dispatch_matrix)

if PRINT_METHODS:

    # determine required member quantity from framing_group quantity_type
    print("\nRequired MEMBER quantity for each section_group (doesn't include transverse laminations):")
    print(mediator.section_group_quantity_df)

    # determine required member length from quantity + frame_group length
    print("\n Required board length for each section_group (includes transverse laminations):")
    print(mediator.section_group_required_length_df)


##############################
# SET-BASED DISPATCH: STAGE 1
# -> actions to increase available lengths (make more section groups structurally feasible)
# -> reduce spacing, reduce span, increase transverse lamination
##############################
print("\nSet-based dispatch stage 1: narrow to feasible structural design space")
print("maximum spans per framing group")
print(mediator.section_group_max_span_df)


if PRINT_METHODS:
    # check design span against maximum span
    print('\n check design span against maximum span')
    print(mediator.section_group_span_capacity_ok_as_boolean)

# dispatch lengths are based on the above
mediator.dispatch_section_groups(DispatchStage.STAGE1_STRUCT)
print(mediator.dispatch_matrix)
print(mediator.normalised_dispatch_matrix)
print(mediator.required_lengths_after_dispatch)



##############################
# SET-BASED DISPATCH: STAGE 2
##############################
print("\nSet-based dispatch stage 2: narrow to feasible part length design space")

# if PRINT_METHODS:
print("minimum part sizes per framing group:")
print(mediator.structure.framing_group_min_part_lengths)

mediator.dispatch_section_groups(DispatchStage.STAGE2_MIN_SIZE)
print(mediator.dispatch_matrix)
print(mediator.normalised_dispatch_matrix)
print(mediator.required_lengths_after_dispatch)


##############################
# SET-BASED DISPATCH: STAGE 3
##############################
#mediator.dispatch_section_groups(DispatchStage.STAGE3_USER)
#needed? wait for larger example
if CASE=="case_2":
    mediator.THRESHOLD_SCORE = 7.8

mediator.dispatch_section_groups(DispatchStage.STAGE3_FINAL_ALLOCATION)
print(mediator.dispatch_matrix)
print(mediator.normalised_dispatch_matrix)
print(mediator.required_lengths_after_dispatch)

# determine required member length from quantity + frame_group length
print("\n Member locations for each section group")
print(mediator.member_location_df)
print(mediator.dispatch_results_summary_df)
print(mediator.dispatch_final_assignment)


if EXPORT_ME:
    results_dir = f"{IO_FOLDERS.RESULTS_DISPATCH.value}{CASE}"
    mediator.export_dispatch(results_dir)

