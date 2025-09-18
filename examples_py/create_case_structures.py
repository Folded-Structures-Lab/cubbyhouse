
import json
import jsonpickle
import copy
from cubbyhouse.utils import IO_FOLDERS
from cubbyhouse.array import DiscreteJoints
from cubbyhouse.building import (
    FramingGroup,
    DeckingParams,
    DeckJoistParams,
    FloorBearerParams,
    WallStudParams,
    JambStudParams,
    LintelParams,
    TopPlateParams,
    BottomPlateParams,
    RafterParams,
    # RoofBattenParams,
    TimberFramedStructure
    # DeckJoistParams
)

####################
# Create Case JSONS
####################

STRUCTURE_CASE_NAMES = ["case_1", "case_2", "case_3"]

for STRUCTURE_CASE_NAME in STRUCTURE_CASE_NAMES:

    if STRUCTURE_CASE_NAME in ["case_1"]:   
        x_extent = 5000
        y_extent = 2300
        deck_design_span = 450
        joist_design_span = 1200 
        bearer_design_span = 1600

        deck_group = FramingGroup(
            "F1",
            length=x_extent,
            width=y_extent,
            design_parameters=DeckingParams(
                design_span=deck_design_span, span_type="continuous"
            ),
            max_parts_per_length=3
        )

        joist_group = FramingGroup(
            "F2",
            length=y_extent,
            width=x_extent,
            design_parameters=DeckJoistParams(
                design_span=joist_design_span,
                span_type="continuous",
                joist_spacing=deck_design_span, #~floor load width
            )
        )

        bearer_group = FramingGroup(
            "F3",
            length=x_extent,
            width=y_extent,
            design_parameters=FloorBearerParams(
                design_span=bearer_design_span,
                span_type="single",
                floor_load_width=joist_design_span,
            ),
            max_parts_per_length=3
        )
        # update framing group joints
        deck_group.update_framing_group_joints(joints_from="design_span")
        joist_group.update_framing_group_joints(joints_from="design_span")
        bearer_group.update_framing_group_joints(joints_from="design_span")
        bearer_group.update_transverse_laminations(2)
        structure = TimberFramedStructure.from_list([deck_group, joist_group, bearer_group])



    if STRUCTURE_CASE_NAME in ["case_2", "case_3"]:   
        house_front_height = 2400
        house_back_height = 2400
        roof_load_width = 1500 
        roof_height=2700
        house_length = 6000
        rafter_spacing = 900  
        stud_spacing_front_wall = 600
        stud_spacing_back_wall=600
        stud_spacing_side_wall = 600
        front_wall_opening = 900
        roof_type="sheet"
        tie_down_spacing = 0
        joist_spacing = 600
        roof_mass = 10
        # batten_spacing=1200

        house_width = roof_load_width*2
        roof_span = (house_width**2 + (roof_height-house_back_height)**2)**0.5
        
        # top_plates_lam_x = 3
        top_plates_lam_y = 2 

        # bot_plates_lam_x = 3
        bot_plates_lam_y = 2 



        front_wall_A = FramingGroup(
            "F1", #front wall
            length=house_front_height,
            width=(house_length-front_wall_opening)*2/3,
            design_parameters=WallStudParams(
                design_span=house_front_height, 
                span_type="single",
                rafter_spacing=rafter_spacing,
                roof_type=roof_type,
                roof_load_width=roof_load_width,
                stud_spacing=stud_spacing_front_wall                
            ),
            max_parts_per_length=1
        )

        front_jamb = FramingGroup(
            "F2",
            length=house_front_height,
            width=front_wall_opening,
            design_parameters=JambStudParams(
                design_span=house_front_height, 
                span_type="single",
                # rafter_spacing=rafter_spacing,
                roof_type=roof_type,
                opening_width=front_wall_opening,
                roof_load_width=roof_load_width,
                # stud_spacing=stud_spacing_front_wall                
            ),
            max_parts_per_length=1
        )

        lintel = FramingGroup(
            "F3",
            length=front_wall_opening,
            width=1,
            design_parameters=LintelParams(
                design_span=front_wall_opening, 
                span_type="single",
                rafter_spacing=rafter_spacing,
                roof_type=roof_type,
                roof_load_width=roof_load_width,         
            ),
            max_parts_per_length=1
        )

        front_wall_B = FramingGroup(
            "F4", #front wall
            length=house_front_height,
            width=(house_length-front_wall_opening)*1/3,
            design_parameters=WallStudParams(
                design_span=house_front_height, 
                span_type="single",
                rafter_spacing=rafter_spacing, 
                roof_type=roof_type,
                roof_load_width=roof_load_width,
                stud_spacing=stud_spacing_front_wall           
            ),
            max_parts_per_length=1
        )

        side_wall_A = FramingGroup(
            "F5", #side wall
            length=house_back_height,
            width=house_width,
            design_parameters=WallStudParams(
                design_span=house_back_height, 
                span_type="single",
                rafter_spacing=rafter_spacing, #? not sure what this is used for -> if RLW =0 though should be irrelevant
                roof_type=roof_type, #ditto
                roof_load_width=0,
                stud_spacing=stud_spacing_side_wall                
            ),
            max_parts_per_length=2,
            transverse_laminations=2
        )


        back_wall = FramingGroup(
            "F6", #back wall
            length=house_back_height,
            width=house_length,
            design_parameters=WallStudParams(
                design_span=house_back_height, 
                span_type="single",
                rafter_spacing=rafter_spacing,
                roof_type=roof_type,
                roof_load_width=roof_load_width,
                stud_spacing=stud_spacing_back_wall                
            ),
            max_parts_per_length=1
        )

        side_wall_B = copy.deepcopy(side_wall_A)
        side_wall_B.name = "F7"


        #############
        # TOP PLATES
        #############

        top_plate_front = FramingGroup(
            "F8", #top_plate
            length=house_length,
            width=1,
            design_parameters=TopPlateParams(
                design_span=stud_spacing_front_wall, 
                span_type="continuous",
                rafter_spacing=rafter_spacing,
                roof_type=roof_type, #ditto
                roof_load_width=roof_load_width,    
                tie_down_spacing = tie_down_spacing    
            ),
            max_parts_per_length=4,
            transverse_laminations=top_plates_lam_y
        )


        top_plate_side_A = FramingGroup(
            "F9", #top_plate
            length=house_width,
            width=1,
            design_parameters=TopPlateParams(
                design_span=stud_spacing_side_wall, 
                span_type="continuous",
                rafter_spacing=rafter_spacing, #needed?
                roof_type=roof_type,  #needed?
                roof_load_width=0,    
                tie_down_spacing = tie_down_spacing    
            ),
            max_parts_per_length=2,
            transverse_laminations=top_plates_lam_y
        )


        top_plate_back = FramingGroup(
            "F10", #top_plate
            length=house_length,
            width=1,
            design_parameters=TopPlateParams(
                design_span=stud_spacing_back_wall, 
                span_type="continuous",
                rafter_spacing=rafter_spacing,
                roof_type=roof_type, #ditto
                roof_load_width=roof_load_width,    
                tie_down_spacing = tie_down_spacing    
            ),
            max_parts_per_length=4,
            transverse_laminations=top_plates_lam_y
        )

        top_plate_side_B = copy.deepcopy(top_plate_side_A)
        top_plate_side_B.name = "F11"
        
        #############
        # BOT PLATES
        #############


        bot_plate_front = FramingGroup(
            "F12", #bot_plate
            length=house_length,
            width=1,
            design_parameters=BottomPlateParams(
                design_span=joist_spacing, 
                span_type="continuous",
                rafter_spacing=rafter_spacing,
                roof_type=roof_type, 
                roof_load_width=roof_load_width,    
                # tie_down_spacing = tie_down_spacing    
            ),
            max_parts_per_length=4,
            transverse_laminations=bot_plates_lam_y
        )


        bot_plate_side_A = FramingGroup(
            "F13", 
            length=house_width,
            width=1,
            design_parameters=BottomPlateParams(
                design_span=0,  #note - parallel to joists
                span_type="continuous",
                rafter_spacing=rafter_spacing, #needed?
                roof_type=roof_type,  #needed?
                roof_load_width=0,       
            ),
            max_parts_per_length=2,
            transverse_laminations=bot_plates_lam_y
        )


        bot_plate_back = FramingGroup(
            "F14", #top_plate
            length=house_length,
            width=1,
            design_parameters=BottomPlateParams(
                design_span=joist_spacing, 
                span_type="continuous",
                rafter_spacing=rafter_spacing,
                roof_type=roof_type, #ditto
                roof_load_width=roof_load_width,    
            ),
            max_parts_per_length=4,
            transverse_laminations=bot_plates_lam_y
        )

        bot_plate_side_B = copy.deepcopy(bot_plate_side_A)
        bot_plate_side_B.name = "F15"
        


        rafters_A = FramingGroup(
            "F16", #rafter
            length=int(roof_span/2), 
            width=house_length,
            design_parameters=RafterParams(
                design_span=int(roof_span/2), 
                span_type="single",
                rafter_spacing=rafter_spacing,
                roof_mass=roof_mass, #ditto    
            ),
            max_parts_per_length=1,
            transverse_laminations=1
        )

        rafters_B = copy.deepcopy(rafters_A)
        rafters_B.name = "F17"
        


        # battens_A = FramingGroup(
        #     "F18", #rafter
        #     length=house_length, 
        #     width=int(roof_span/2),
        #     design_parameters=RoofBattenParams(
        #         design_span=rafter_spacing, 
        #         span_type="continuous",
        #         roof_type=roof_type,
        #         batten_spacing=batten_spacing,  
        #     ),
        #     max_parts_per_length=3,
        #     transverse_laminations=1
        # )

        # battens_B = copy.deepcopy(battens_A)
        # battens_B.name = "F19"


        # update framing group joints
        framing_groups = [
            front_wall_A, front_jamb, lintel, front_wall_B,
            side_wall_A, 
            back_wall,
            side_wall_B,
            top_plate_front, top_plate_side_A, top_plate_back, top_plate_side_B,
            bot_plate_front, bot_plate_side_A, bot_plate_back, bot_plate_side_B,
            rafters_A, rafters_B
        ]

        for framing_group in framing_groups:
            if framing_group not in [bot_plate_side_A, bot_plate_side_B, side_wall_A, side_wall_B]:
                framing_group.update_framing_group_joints(joints_from="design_span")
            elif framing_group in [bot_plate_side_A, bot_plate_side_B]:
                framing_group.discrete_joints= DiscreteJoints.from_support_span(framing_group.length, stud_spacing_side_wall)
            else:
                framing_group.discrete_joints=DiscreteJoints.from_support_span(framing_group.length, house_back_height/3)

        # if STRUCTURE_CASE_NAME == "case_3":
        #     house_framing_groups = []
        #     num_houses = 8
        #     suffix_houses = [chr(x+65) for x in range(num_houses)]
        #     for suffix in range(num_houses):
                
        #         for framing_group in framing_groups:
        #             fg = copy.deepcopy(framing_group)
        #             fg.name = f"{fg.name}-{suffix}"
        #             house_framing_groups.append(fg)
        #     framing_groups = house_framing_groups

        structure = TimberFramedStructure.from_list(framing_groups)



    #Export
    export_name = IO_FOLDERS.STRUCTURE.value + STRUCTURE_CASE_NAME + '_structure.json'
    with open(export_name, 'w') as f:
        json.dump(jsonpickle.encode(structure), f)