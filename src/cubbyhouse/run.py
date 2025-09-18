import json

# from cubbyhouse.supports import SpanContinuity, DiscreteSupport, PartCombinationSet
# from cubbyhouse.combine import BoardPartSet, AssignmentSet, AssignInventory
from cubbyhouse.boardstock import BoardStock
from cubbyhouse.utils import get_stock_file_from_case_id
from cubbyhouse.building import DeckingParams, DeckJoistParams,FramingGroup, design_param_class_mapper
# from cubbyhouse.supports import DiscreteSupport  # , PartCombinationSet


def run_me():
    return "Hi!"

def board_stock_from_case_id(case_id):
    case_stock = get_stock_file_from_case_id(case_id)
    board_stock = BoardStock.from_element_csv(case_stock)

    return {
        # "joint_ordinates": discrete_support.ordinates,
        # "unique_lengths": part_combos.unique_lengths,
        "element_df": board_stock.element_df.to_json(orient="records"),
        "element_group_df": board_stock.element_group_df.to_json(orient="records"),
    }

def gh_decking_design_parameters(span_type, design_span):
    design_params = DeckingParams(span_type=span_type, design_span=design_span)
    return design_params.to_json()

def gh_deck_joist_design_parameters(span_type, design_span, joist_spacing):
    design_params = DeckJoistParams(span_type=span_type, design_span=design_span, joist_spacing=joist_spacing)
    return design_params.to_json()



def gh_framing_group(name:str, length:float, width:float, design_parameters):
    #print(type(design_parameters))

    dps_json = json.loads(design_parameters)
    design_param_class = design_param_class_mapper(dps_json["framing_type"])

    fg = FramingGroup(
        name= name, 
        length=length,
        width=width,
        design_parameters=design_param_class.from_json(design_parameters)
        )
    return fg.to_json()

# dps = gh_decking_design_parameters("single", 450)
# fg = gh_framing_group('h1', 5010, 2030, dps)
# print(fg)

# def run_me_2(target_length, max_member_span, min_member_span, max_parts):
#     continuity = SpanContinuity.DOUBLE
#     print("run_me_2 running")
#     # )
#     # return {"ordinates": a.ordinates}
#     # return f"running_OK target_length={target_length}"
#     # get discrete support locations
#     # target_length = 5970
#     # max_member_span = 450
#     # min_member_span = 150
#     # max_parts = 3
#     # #inv_fname = "case1_inventory.csv"

#     discrete_support = DiscreteSupport(target_length, max_member_span, min_member_span)
#     part_combos = PartCombinationSet(discrete_support, continuity, max_parts=max_parts)
#     #part_combos.unique_lengths
#     return {
#         "joint_ordinates": discrete_support.ordinates,
#         "unique_lengths": part_combos.unique_lengths,
#         "df": part_combos.df.to_json(orient="index"),
#     }
#     # return discrete_support  # , part_combos


# a = run_me_2(4000)
# print(str(a))
