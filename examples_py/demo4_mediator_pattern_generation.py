from cubbyhouse.mediator import CubbyhouseMediator, DispatchStage
from cubbyhouse.utils import get_stock_file_from_case_id, get_structure_from_case_id, IO_FOLDERS
from cubbyhouse.boardstock import BoardStock
# import pandas as pd


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
# SET-BASED DISPATCH: STAGE 0-3
################################
print("\nSet-based dispatch D3:")
if CASE=="case_2":
    mediator.THRESHOLD_SCORE = 7.8
mediator.dispatch_section_groups(DispatchStage.STAGE3_FINAL_ALLOCATION)
print(mediator.dispatch_matrix)
print(mediator.normalised_dispatch_matrix)

##############################
# PART ASSIGNMENT
##############################
#from cubbyhouse.combine import  AssignmentSet, AssignInventory

#create members including member jointing patterns
mediator.create_members()

if PRINT_ME: 
    for member_name, member in mediator.structure.members.items():    

        print(f"\n Unique Parts for Member {member_name}:")
        print(member.jointing_patterns.unique_part_names_used_in_jointing_patterns)

        print(f"\n J_m Parts Used Per Jointing Patterns for {member_name}:")
        print(member.jointing_patterns.part_count_df)

        print(f"\n m_m Members Used Per Jointing Patterns for {member_name}:")
        print(member.jointing_patterns.members_per_joint_pattern)

    #Jointing Pattern Pattern Summary
    print(f"\n n_P,m Possible parts per member group: \n {mediator.n_Pm}")
    print(f"\n n_J,m Jointing patterns per member group: \n {mediator.n_Jm}")


    #GLOBAL MEMBER VECTOR AND JOINTING PATTERN MATRICES
    print(f'\n Member Demand Vector Q_m: \n {mediator.Q_m}')
    print(f'\n Jointing Pattern Member Supply Matrix [M]: \n {mediator.M_matrix}')
    print(f'\n Jointing Pattern Part Demand Matrix [J]: \n {mediator.J_matrix}')



mediator.create_element_group_cutting_patterns(pattern_type = 'no_usable_residual')
if PRINT_ME:

    for section_group_name, cutting_patterns in mediator.cutting_patterns.items(): 
        print(f"\n Element cutting patterns for {section_group_name}:")
        print(cutting_patterns.df)

        print(f"\n C_c Parts Used Per Cutting Pattern for {section_group_name}:")
        print(cutting_patterns.part_count_df)
        
        print(f"\n e_c Elements Used Per Cutting Pattern for {section_group_name}:")
        print(cutting_patterns.element_count_df)


    #Part Summary
    print("\n Unique Parts for Structure:")
    structure_parts = mediator.structure.parts
    print(f'\n n_Pk Unique parts per section group: \n {mediator.n_Pk}')

    #Cutting Pattern Summary
    print(f"\n n_Ci Cutting patterns per element group: \n {mediator.n_Ci}")
    print(f"\n n_Pi Parts per element group: \n {mediator.n_Pi}")
    print(f"\n n_Pk Parts per section group: \n {mediator.n_Pk}")

    # GLOBAL ELEMENT VECTOR AND CUTTING PATTERN MATRICES
    print(f'\n Cutting Pattern Part Supply Matrix [C]: \n {mediator.C_matrix}')
    print(f'\n Cutting Pattern Element Demand Matrix [E]: \n {mediator.E_matrix}')
    print(f'\n Cutting Pattern Waste Vector [w]:\n {mediator.w_vector}')
    print(f'\n Element Supply Vector: \n {mediator.Q_e}')



if EXPORT_ME:
    results_dir = f"{IO_FOLDERS.RESULTS.value}{CASE}"
    mediator.export_matrices(results_dir)
    mediator.export_parts(results_dir)

    #Export Stock
    mediator.export_elements(results_dir)
    mediator.export_element_groups(results_dir)
    mediator.export_section_groups(results_dir)

    #export Structure
    mediator.export_member_groups(results_dir)
    mediator.export_members(results_dir)

    #export Patterns
    mediator.export_cutting_patterns(results_dir)
    mediator.export_jointing_patterns(results_dir)
