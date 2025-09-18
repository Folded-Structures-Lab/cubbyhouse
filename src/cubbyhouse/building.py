from __future__ import annotations

from dataclasses import asdict, dataclass, field
from dataclasses_json import config, dataclass_json

import pandas as pd
import math
from numpy import arange
from enum import Enum
from cubbyhouse.array import DiscreteJoints, MemberJointingPatterns#, SpanContinuity
from cubbyhouse.utils import FRAMING_TYPE
from cubbyhouse.boardstock import BoardStock


class QUANTITY_TYPE(Enum):
    UNIT = "unit"
    SPACED = "spaced"
    FILL = "fill"
    EXPLICIT = "explicit"

def design_param_class_mapper(framing_type: str | FRAMING_TYPE):
    mapper = {
        FRAMING_TYPE.DECKING: DeckingParams,
        FRAMING_TYPE.DECK_JOIST: DeckJoistParams,
        FRAMING_TYPE.FLOOR_BEARER: FloorBearerParams
    }
    return mapper[framing_type]




@dataclass_json
@dataclass
class FramingDesignParameters:
    span_type: str = "single"
    design_span: float = 0
    framing_type: FRAMING_TYPE | None = None

    quantity_type: QUANTITY_TYPE | None = None
    spacing: float | None = None

    def as_dict(self):
        return asdict(self)

    def spacing_calculation(self):
        raise NotImplementedError
    
    def member_quantity(self,**kwargs):
        if self.quantity_type == QUANTITY_TYPE.UNIT:
            qty = 1
        elif self.quantity_type == QUANTITY_TYPE.SPACED:
            if 'width' in kwargs:
                if self.spacing is not None: 
                    qty = math.ceil(kwargs['width']/self.spacing) +1
                else: 
                    raise ValueError('No spacing given for design parameter quantity calculation')
            else: 
                raise ValueError('No width given for design parameter quantity calculation')
        elif self.quantity_type == QUANTITY_TYPE.FILL:
            if 'width' in kwargs and 'board_width' in kwargs:
                #TODO - PUT MIN SPACING ELSEWHERE
                min_spacing = 3
                effective_board_width = kwargs['board_width']+min_spacing
                qty = math.ceil(kwargs['width']/effective_board_width)
        elif self.quantity_type== QUANTITY_TYPE.EXPLICIT:
            qty  = self.explicit_quantity
        else:
            raise ValueError()
        return qty

    def member_locations(self,member_quantity, **kwargs):
        if self.quantity_type in [QUANTITY_TYPE.UNIT, QUANTITY_TYPE.EXPLICIT]:
            locs=[0]
        if self.quantity_type==QUANTITY_TYPE.SPACED:
            locs = arange(0,kwargs['width'],self.spacing).tolist()
            if locs[-1]!=kwargs['width']:
                locs.append(kwargs['width'])
            # locs[0]= locs[0]+
        if self.quantity_type == QUANTITY_TYPE.FILL:
            # locs=[0,0,0]
            #TODO - PUT MIN SPACING ELSEWHERE
            min_spacing = 3
            effective_board_width = kwargs['board_width']+min_spacing
            # np.cumsum()
            locs = arange(0,kwargs['width'],effective_board_width).tolist()
            # locs = [x+kwargs['board_width']/2 for x in locs]
            ...
        

        return locs


@dataclass_json
@dataclass
class DeckingParams(FramingDesignParameters):
    

    def __post_init__(self):
        # self.SPACING_FROM_DESIGN_PARAMETERS = False
        self.framing_type = FRAMING_TYPE.DECKING
        self.quantity_type = QUANTITY_TYPE.FILL

@dataclass_json
@dataclass
class DeckJoistParams(FramingDesignParameters):
    joist_spacing: float = 450

    def __post_init__(self):
        self.framing_type = FRAMING_TYPE.DECK_JOIST
        self.quantity_type = QUANTITY_TYPE.SPACED
        self.spacing = self.joist_spacing


@dataclass_json
@dataclass
class FloorBearerParams(FramingDesignParameters):
    floor_load_width: float = 1200
    roof_load_width: float = 0

    def __post_init__(self):
        self.framing_type = FRAMING_TYPE.FLOOR_BEARER
        self.quantity_type = QUANTITY_TYPE.SPACED
        self.spacing = self.floor_load_width


@dataclass_json
@dataclass
class LintelParams(FramingDesignParameters):
    roof_type: str = "sheet"
    rafter_spacing: float = 600
    roof_load_width: float = 7500

    def __post_init__(self):
        self.framing_type = FRAMING_TYPE.LINTEL
        self.quantity_type = QUANTITY_TYPE.UNIT


@dataclass_json
@dataclass
class RafterParams(FramingDesignParameters):
    roof_mass: float = 40
    rafter_spacing: float = 450

    def __post_init__(self):
        self.framing_type = FRAMING_TYPE.RAFTER
        self.quantity_type = QUANTITY_TYPE.SPACED
        self.spacing = self.rafter_spacing


@dataclass_json
@dataclass
class BottomPlateParams(FramingDesignParameters):
    rafter_spacing: float = 450
    roof_type: str = "sheet"
    roof_load_width: float = 0


    def __post_init__(self):
        self.framing_type = FRAMING_TYPE.BOTTOM_PLATE
        self.quantity_type = QUANTITY_TYPE.UNIT

@dataclass_json
@dataclass
class TopPlateParams(FramingDesignParameters):
    rafter_spacing: float = 450
    roof_type: str = "sheet"
    roof_load_width: float = 0
    tie_down_spacing: float = 0

    def __post_init__(self):
        self.framing_type = FRAMING_TYPE.TOP_PLATE
        self.quantity_type = QUANTITY_TYPE.UNIT

@dataclass_json
@dataclass
class WallStudParams(FramingDesignParameters):

    rafter_spacing: float = 450
    roof_type: str = "sheet"
    roof_load_width: float = 0
    stud_spacing: float = 600

    def __post_init__(self):
        self.framing_type = FRAMING_TYPE.WALL_STUD
        self.quantity_type = QUANTITY_TYPE.SPACED
        self.spacing = self.stud_spacing





@dataclass_json
@dataclass
class JambStudParams(FramingDesignParameters):
    roof_type: str = "sheet"
    roof_load_width: float = 0
    opening_width: float = 600

    def __post_init__(self):
        self.framing_type = FRAMING_TYPE.JAMB_STUD
        self.quantity_type=QUANTITY_TYPE.SPACED
        self.spacing = self.opening_width



@dataclass_json
@dataclass
class RoofBattenParams(FramingDesignParameters):
    batten_spacing: float = 600
    roof_type: str = "sheet"

    def __post_init__(self):
        self.framing_type = FRAMING_TYPE.ROOF_BATTEN
        self.quantity_type = QUANTITY_TYPE.SPACED
        self.spacing = self.batten_spacing



@dataclass_json
@dataclass
class PartsOnlyParams(FramingDesignParameters):
    explicit_quantity: int = 0

    def __post_init__(self):
        # self.SPACING_FROM_DESIGN_PARAMETERS = False
        self.framing_type = FRAMING_TYPE.PART_ONLY
        self.quantity_type = QUANTITY_TYPE.EXPLICIT


@dataclass_json
@dataclass
class FramingGroup:
    """TODO"""

    name: str
    length: float
    width: float

    design_parameters: FramingDesignParameters

    span_table: pd.DataFrame | None = field(default = None, metadata=config(exclude=lambda x: x is None))

    transverse_laminations: int = 1  # transverse board laminations

    discrete_joints: DiscreteJoints | None = field(default = None, metadata=config(exclude=lambda x: x is None))
    max_parts_per_length: int | None = field(default = 1, metadata=config(exclude=lambda x: x is None))
    


    def __post_init__(self):
        if self.discrete_joints is None:
            self.update_framing_group_joints()

    def add_span_table(self, span_table_df: pd.DataFrame):
        """localised a framing_type span table from the timber_db, filtering for the current design parameters"""
        design_parameters = self.design_parameters.as_dict()
        for key, value in design_parameters.items():
            if key in span_table_df.columns:
                span_table_df = span_table_df[span_table_df[key] == value]
        self.span_table = span_table_df

    def update_framing_group_joints(self, joints_from: str = "length") -> None:
        if joints_from == "length":
            self.discrete_joints = DiscreteJoints.from_length(self.length)
        if joints_from == "design_span":
            self.discrete_joints = DiscreteJoints.from_support_span(self.length, self.design_parameters.design_span)

    def update_transverse_laminations(self, transverse_lams: int):
       self.transverse_laminations = transverse_lams

    def calc_quantities(self, stock: BoardStock):
        ...
        #TODO iterate over boardstock
    
        # self.design_parameters.member_quantity()
        quantities = pd.DataFrame(index=stock.section_group_df.index,columns=['qty'])
        if self.design_parameters.quantity_type == QUANTITY_TYPE.UNIT:
            qty = self.design_parameters.member_quantity()
            quantities['qty'] = qty
        elif self.design_parameters.quantity_type == QUANTITY_TYPE.SPACED:
            qty = self.design_parameters.member_quantity(**{"width":self.width})
            quantities['qty'] = qty
        elif self.design_parameters.quantity_type == QUANTITY_TYPE.FILL:
            for ind, row in stock.section_group_df.iterrows():  
                element_group_id = row.element_group_indices[0]
                element_group = stock.element_group_df.loc[element_group_id, :]
                board_width = max(element_group["D"], element_group["B"])
                qty = self.design_parameters.member_quantity(**{"width":self.width, "board_width": board_width})
                quantities.loc[ind]=qty
            # return 1
        elif self.design_parameters.quantity_type== QUANTITY_TYPE.EXPLICIT:
            qty = self.design_parameters.member_quantity()
            quantities['qty'] = qty

        return quantities

    def calc_locations(self, stock: BoardStock):
        
        member_locations = pd.DataFrame(index=stock.section_group_df.index,columns=['qty'])
        quantities = self.calc_quantities(stock=stock)

        for ind, row in stock.section_group_df.iterrows(): 
            element_group_id = row.element_group_indices[0]
            element_group = stock.element_group_df.loc[element_group_id, :]
            board_width = max(element_group["D"], element_group["B"])
            member_qty = quantities.loc[ind].values[0]
            # if self.design_parameters.quantity_type==QUANTITY_TYPE.
            cur_locations= self.design_parameters.member_locations(member_qty, **{"width":self.width, "board_width": board_width})
            member_locations.at[ind,'qty']=cur_locations


        return member_locations


    @property
    def framing_type(self) -> FRAMING_TYPE:
        return self.design_parameters.framing_type
    
    @property
    def total_element_length(self) -> float:
        """total required element length"""
        return self.length * self.element_count

    @property
    def span_type(self):
        return self.design_parameters.span_type 

    @property
    def min_part_length(self) -> float:
        if self.max_parts_per_length == 1:
            return self.length
        else:
            return self.discrete_joints.min_part_length(self.span_type)

    @property
    def span_table_mapper(self) -> pd.DataFrame:
        """function to reduce span_table from additional element design attributes"""
        return self.span_table

    @property
    def data_dict(self)-> dict:
        return {
            "name": self.name,
            "length": self.length,
            "width": self.width,
            "lambda_y": self.transverse_laminations,
            "framing_type": self.design_parameters.framing_type.value,
            "design_span": self.design_parameters.design_span,
            "span_type": self.design_parameters.span_type,
            "quantity_type": self.design_parameters.quantity_type.value,
            "lambda_x": self.max_parts_per_length,
            "min_part_length": self.min_part_length,
            "joint_ordinates": self.discrete_joints.ordinates
        }


    # raise NotImplementedError



@dataclass
class Member:
    ...
    name:str
    jointing_patterns: MemberJointingPatterns | None = field(default = None, metadata=config(exclude=lambda x: x is None))
    qty: int | None = None
    framing_group_name: str | None = None
    #members_per_joint_pattern: pd.DataFrame | None = None   

    # @property
    # def possible_part_lengths(self) -> list[float]:
    #     """set of possible (unique) part lengths from part combination set"""
    #     return self.assembly_patterns.unique_lengths

    def solve_jointing_patterns(self, discrete_joints, span_type, max_parts_per_length, transverse_laminations: int):
        """solve member assembly patterns (part combinations)"""
        self.jointing_patterns = MemberJointingPatterns(
            discrete_joints,
            span_type,
            max_parts=max_parts_per_length,
        )
        self.jointing_patterns.rename_patterns(self.name)
        self.jointing_patterns.update_for_transverse_laminations(transverse_laminations)
        member_joint_patterns = pd.DataFrame(index=self.jointing_patterns.part_count_df.index, columns = [self.name])
        member_joint_patterns[:]=1
        self.jointing_patterns.members_per_joint_pattern = member_joint_patterns



@dataclass
class TimberFramedStructure:
    framing_groups: dict[str, FramingGroup]
    members: dict[str, Member] | None = None
    parts: pd.DataFrame | None = None

    def add_member(self,member_name: str, member: Member) -> None:
        #TODO - check if member exists
        if self.members is None:
            self.members = {}
        self.members[member_name] = member

    def add_parts(self, section_group_name: str, member: Member)->None:
        col_names = ['part_length', 'section_group']
        new_parts = member.jointing_patterns.unique_parts.copy()
        new_parts = new_parts.to_frame(name=col_names[0])
        new_parts[col_names[1]] = section_group_name
        if self.parts is None:
            self.parts = pd.DataFrame(columns=col_names)
            self.parts = new_parts
        else:        
            relevant_existing_parts = self.parts[self.parts['section_group']==section_group_name]

            
            rename_dict = {}
            new_names = []
            existing_part_count = self.part_count
            for ind, row in new_parts.iterrows():
                if row['part_length'] in list(relevant_existing_parts['part_length'].values):
                    #use existing name
                    new_names.append(relevant_existing_parts[relevant_existing_parts['part_length']==row['part_length']].index.values[0])
                else:
                    #create new name
                    new_names.append(f'P{existing_part_count}')
                    existing_part_count = existing_part_count+ 1
            
            
            #save new names 
            new_parts['new_names'] = new_names
            #dict mapping old names to new names
            rename_dict = dict(zip(list(new_parts.index), list(new_parts['new_names'])))
            #set index on new parts to new names
            new_parts.index=new_parts['new_names']
            new_parts = new_parts[['part_length', 'section_group']]
            #combine with existing parts
            result_df = pd.concat([self.parts, new_parts])
            result_df = result_df[~result_df.index.duplicated(keep='first')]  # Keep the first occurrence
            self.parts=result_df

            #rename member parts
            if len(rename_dict)>0:
                if len(new_parts) > 1:
                    #squeeze makes a series from dataframe if len > 1
                    new_parts_as_series = new_parts[['part_length']].squeeze()
                else:
                    #squeeze makes a scalar from dataframe if len = 1
                    #make series manually
                    new_parts_as_series = pd.Series(new_parts['part_length']) 
                    new_parts_as_series.index.name=None

                member.jointing_patterns.rename_parts(new_parts_as_series, rename_dict)
            ...


    @property
    def framing_group_names(self) -> list[str]:
        """list of framing group names"""
        return list(self.framing_groups.keys())
    
    @property
    def framing_group_df(self) -> pd.DataFrame:
        all_data = []
        for _, framing_group in self.framing_groups.items():
            all_data.append(framing_group.data_dict)
        df = pd.DataFrame(all_data)
        df.set_index("name", inplace=True)
        return df
            

    @property
    def member_names(self) -> list[str]:
        """list of member names"""
        return list(self.members.keys())

    @property
    def part_names(self) -> list[str]:
        """list of part names"""
        return list(self.parts.index)


    @property
    def framing_group_types(self) -> list[FRAMING_TYPE]:
        """list of framing group types"""
        return [group.framing_type for group in self.framing_groups.values()]

    @property
    def framing_group_min_part_lengths(self) -> dict:
        min_part_lengths = {key: val.min_part_length for key, val in self.framing_groups.items()}
        return min_part_lengths

    @property
    def framing_group_design_spans_df(self) -> pd.Series:
        """datafram series of framing group design spans"""
        data = {
            group.name: group.design_parameters.design_span
            for group in self.framing_groups.values()
        }
        return pd.Series(data)

    @classmethod
    def from_list(cls, framing_groups: list[FramingGroup]):
        framing_groups = {group.name: group for group in framing_groups}
        return cls(framing_groups=framing_groups)


    @property
    def member_quantities(self) -> pd.Series:
        data = {
            member_name: member.qty
            for member_name, member in self.members.items()
        }
        df = pd.Series(data).to_frame()
        df.columns=['qty']
        return df
    
    @property
    def part_count(self) -> pd.Series:
        return len(self.parts)
    


# @dataclass
# class DeckingFramingGroup(FramingGroup):
#     """FramingGroup implementation for decking boards"""

#     @property
#     def span_table_mapper(self) -> pd.DataFrame:
#         return self.span_table


# @dataclass
# class StudFramingGroup(FramingGroup):
#     """FramingGroup implementation for common studs (upper/single story, external wall)"""

#     roof_type: str = "sheet"
#     rafter_spacing: float = 0
#     RLW: float = 3000

#     @property
#     def span_table_mapper(self) -> pd.DataFrame:
#         # map element design attributes to span table
#         small_span_table = self.span_table.query(
#             """ stud_spacing == @self.spacing and \
#             roof_type == @self.roof_type and \
#             rafter_spacing == @self.rafter_spacing and \
#             max_RLW >= @self.RLW"""
#         )
#         # rename stud_height to max_span
#         small_span_table.rename(columns={"stud_height": "max_span"}, inplace=True)
#         # keep max_span row for each unique 'size' and 'grade'
#         idx_max_spans = small_span_table.groupby(["size", "grade"])["max_span"].idxmax()
#         return small_span_table.loc[idx_max_spans, ["size", "grade", "max_span"]]


# @dataclass
# class DoubleStudFramingGroup(StudFramingGroup):
#     ...
