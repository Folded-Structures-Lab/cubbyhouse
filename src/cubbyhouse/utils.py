"""TODO"""

from enum import Enum
import json
import jsonpickle
from pathlib import Path


class OUTPUT_CSVS(Enum):
    JOINT_LOCATIONS =  "1_joint_locations.csv"
    PART_COMBOS=  "2_part_combos.csv"
    BOARD_INVENTORY = "3_board_inventory.csv"
    BOARD_PART_COMBOS = "4_board_part_combos.csv"
    ASSIGNMENT = "5A_assignment.csv"
    SORTED_ASSIGNMENT = "5B_sorted_assignment.csv"
    FINAL_ASSIGNMENT = "5C_final_assignment.csv"
    

class FRAMING_TYPE(Enum):
    DECKING = "decking"
    DECK_JOIST = "deck_joist"
    FLOOR_BEARER = "floor_bearer"
    LINTEL = "lintel"
    RAFTER = "rafter"
    BOTTOM_PLATE = "bottom_plate"
    TOP_PLATE = "top_plate"
    JAMB_STUD = "jamb_stud"
    WALL_STUD = "wall_stud"
    ROOF_BATTEN = "roof_batten"
    PART_ONLY = "part_only"
    



CUBBYHOUSE_DIR = str(Path(__file__).parent.absolute())
PARENT_DIR = str(Path(__file__).parent.parent.parent.absolute())
# example_dir = str(p1.parent.parent.absolute())


class IO_FOLDERS(Enum):
    SPANTABLES = CUBBYHOUSE_DIR + "\\spantables\\"
    STOCK = CUBBYHOUSE_DIR + "\\eg_stock\\"
    STRUCTURE = CUBBYHOUSE_DIR + "\\eg_structure\\"
    RESULTS_DISPATCH = PARENT_DIR + "\\results\\method_1_dispatch\\"
    RESULTS = PARENT_DIR + "\\results\\"


    # output_dir = output_dir + CASE + "\\"

def get_stock_file_from_case_id(case_id: str):
    return str(IO_FOLDERS.STOCK.value) + case_id + "_inventory.csv"


# def get_stock_from_case_id(case_id: str):
#     file_name = get_stock_file_from_case_id(case_id)
#     board_stock = BoardStock.from_element_csv(file_name)
#     return board_stock



def get_structure_from_case_id(case_id: str):
    import_name = str(IO_FOLDERS.STRUCTURE.value) + case_id + '_structure.json'
    with open(import_name) as f:
        structure_json = json.load(f)

    structure = jsonpickle.decode(structure_json)
    return structure






# def subset_sum_with_repetition(numbers, target_sum):
#     def find_combinations(start, target_sum, current_combination, combinations):
#         if target_sum == 0:
#             combinations.append(current_combination)
#             return
#         if target_sum < 0:
#             return
#         for i in range(start, len(numbers)):
#             find_combinations(i, target_sum - numbers[i], current_combination + \
# [numbers[i]], combinations)

#     combinations = []
#     find_combinations(0, target_sum, [], combinations)
#     return combinations


def subset_sum_with_repetition_and_order(
    numbers: list[int], target_sum: int
) -> list[list[int]]:
    """
    This function finds all possible combinations of the provided numbers that add up to the target
    sum. Repetitions and order are taken into account.

    Args:
        numbers (list[int]): A list of integers.
        target_sum (int): The target sum for which combinations are to be found.

    Returns:
        list[list[int]]: A list of lists, where each inner list represents a combination that
        adds up to the target sum.
    """

    def find_combinations(
        start: int,
        target_sum: int,
        current_combination: list[int],
        combinations: list[list[int]],
    ) -> None:
        """
        This helper function finds all the combinations of numbers that add up to the target sum.

        Args:
            start (int): The starting index for the recursive search.
            target_sum (int): The remaining target sum for which combinations are to be found.
            current_combination (list[int]): The current combination of numbers being constructed.
            combinations (list[list[int]]): The list of valid combinations found so far.
        """
        if target_sum == 0:
            combinations.append(current_combination)
            return
        if target_sum < 0:
            return
        for ind in range(start, len(numbers)):
            find_combinations(
                0,
                target_sum - numbers[ind],
                current_combination + [numbers[ind]],
                combinations,
            )

    combinations = []
    find_combinations(0, target_sum, [], combinations)
    return combinations


# #UTILIY SCRIPT RETRIEVE PART
# partID = 1.025
# board_pos = next_board_any_pos_df[next_board_any_pos_df['part_id']==partID]
# pos_data = pair_df[pair_df['pos_start_end']==board_pos['pos_start_end'].values[0]]
# inv_data = inv_df[inv_df['board_id']==board_pos['board_id'].values[0]]
