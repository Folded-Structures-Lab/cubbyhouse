from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from math import nan
from copy import deepcopy
import pandas as pd

from cubbyhouse.boardstock import BoardStock
from cubbyhouse.building import TimberFramedStructure, Member
from cubbyhouse.utils import FRAMING_TYPE, IO_FOLDERS
from cubbyhouse.combine import ElementCuttingPatterns, AssemblyPatterns
from cubbyhouse.array import MemberJointingPatterns

class DispatchStage(Enum):
    STAGE0_ALL = "stage0_all"
    STAGE1_STRUCT = "stage1_struct"
    STAGE2_MIN_SIZE = "stage2_min_size"
    STAGE3_FINAL_ALLOCATION= "stage3_final_allocation"


@dataclass
class TimberDB:
    span_tables: dict[FRAMING_TYPE, pd.DataFrame] = field(default_factory=dict)

    SPAN_TABLE_MAPPER = {
        FRAMING_TYPE.DECKING: "T_decking.csv",
        FRAMING_TYPE.DECK_JOIST: "T50_deck_joists.csv",
        FRAMING_TYPE.FLOOR_BEARER: "T5_floor_bearer_floor_load_only.csv",
        FRAMING_TYPE.LINTEL: "T17T18_lintel.csv",
        FRAMING_TYPE.RAFTER: "T29_rafter.csv",
        FRAMING_TYPE.BOTTOM_PLATE: "T14_bottom_plate.csv",
        FRAMING_TYPE.TOP_PLATE: "T15T16_top_plate.csv",
        FRAMING_TYPE.JAMB_STUD: "T11_jamb_stud.csv",
        FRAMING_TYPE.WALL_STUD: "T7_wall_stud.csv",
        FRAMING_TYPE.ROOF_BATTEN: "T32_roof_batten.csv",
        FRAMING_TYPE.PART_ONLY: "N/A"
    }

    def add_span_table_from_file(self, framing_type: FRAMING_TYPE):

        if framing_type not in self.SPAN_TABLE_MAPPER.keys():
            raise ValueError(
                f"Framing type {framing_type} not found in SPAN_TABLE_MAPPER"
            )

        file_name = str(IO_FOLDERS.SPANTABLES.value) + self.SPAN_TABLE_MAPPER[framing_type]
        span_table = pd.read_csv(file_name)
        self.span_tables[framing_type] = span_table

    def add_span_table(self,framing_type: FRAMING_TYPE, span_table: pd.DataFrame):
        self.span_tables[framing_type] = span_table



@dataclass
class CubbyhouseMediator:
    """central controller to solve the cubbyhouse design problem"""

    structure: TimberFramedStructure
    stock: BoardStock
    timber_db: TimberDB = field(init=False)

    dispatch_matrix: pd.DataFrame = field(init=False)
    normalised_dispatch_matrix: pd.DataFrame = field(init=False)

    cutting_patterns: Optional[dict[str, ElementCuttingPatterns]] = field(repr=False, default=None)
    assembly_patterns: Optional[dict[str, AssemblyPatterns]] = field(repr=False, default=None)
    #jointing_patterns are created as Member attribute - accessed via property in CubbyhouseMediator

    THRESHOLD_SCORE: int = 1.1
    MEMBER_MULTIPLIER: int = 1

    def __post_init__(self):
        # self.create_dispatch_matsrix(DispatchStage.STAGE0_ALL)
        self.add_span_table_database()
        self.add_framing_group_span_tables()

    def add_span_table_database(self):
        timber_database = TimberDB()
        for framing_group_type in self.structure.framing_group_types:
            if framing_group_type not in timber_database.span_tables.keys():
                if framing_group_type != FRAMING_TYPE.PART_ONLY:
                    #import span tables for normal framing group types
                    timber_database.add_span_table_from_file(framing_group_type)
                else:
                    ...
                    #construct a span table which includes all element group sizes and grade
                    span_table=pd.DataFrame(columns=["size", "grade","span_type","max_span"])
                    span_table[['size','grade',]] = self.stock.section_group_df[['size','grade',]]
                    span_table[["span_type"]]= "single"
                    span_table[["max_span"]]= 1
                    span_table.reset_index(drop=True,inplace=True)
                    timber_database.add_span_table(FRAMING_TYPE.PART_ONLY, span_table)

            print(f"Loaded span table for {framing_group_type}")
        self.timber_db = timber_database

    def add_framing_group_span_tables(self) -> None:
        for _, framing_group in self.structure.framing_groups.items():
            span_table = self.timber_db.span_tables[framing_group.framing_type]
            framing_group.add_span_table(span_table)
            ...

    def dispatch_section_groups(
        self, dispatch_stage: DispatchStage = DispatchStage.STAGE0_ALL
    ) -> None:
        """create a dispatch matrix for the cubbyhouse design problem"""
        framing_cols = self.structure.framing_group_names
        dispatch_df = pd.DataFrame(
            index=self.stock.section_group_df.index, columns=framing_cols
        )

        #Stage 0 -> all lengths
        # Create a new DataFrame with the row indices and L_total
        length_vals = self.stock.section_group_df["L_total"].values
        # Assign new columns with L_total values
        for column in framing_cols:
            dispatch_df[column] = length_vals
        


        # Stage 1
        if dispatch_stage == DispatchStage.STAGE1_STRUCT:
            dispatch_df = self.section_group_span_capacity_dispatch(dispatch_df)
            
        # Stage 2
        if dispatch_stage == DispatchStage.STAGE2_MIN_SIZE:
            dispatch_df = self.section_group_span_capacity_dispatch(dispatch_df)
            dispatch_df = self.section_group_min_part_size_dispatch(dispatch_df)
 
        # Stage 3
        if dispatch_stage == DispatchStage.STAGE3_FINAL_ALLOCATION:
            dispatch_df = self.section_group_span_capacity_dispatch(dispatch_df)
            dispatch_df = self.section_group_min_part_size_dispatch(dispatch_df)
            required_df = self.section_group_required_length_df
            normalised_df = dispatch_df.div(required_df)


            final_dispatch_df = pd.DataFrame(index=dispatch_df.index,columns=dispatch_df.columns).fillna(0)

            unassigned_framing_groups = len(dispatch_df.columns)
            # Step 3: Define a threshold score
            threshold_score = self.THRESHOLD_SCORE

            assigned_framing_groups = []

            while unassigned_framing_groups > 0:
                
                # Step 1: Calculate the average value for each column in normalised_df
                col_avg = normalised_df.mean()
                col_avg.drop(assigned_framing_groups,inplace=True)
                # Step 2: Sort the columns based on their average values (ascending order)
                sorted_cols = col_avg.sort_values().index
                target_fg = sorted_cols[0] #framing group being allocated this iteration
                # print(target_fg)
                unassigned_framing_groups = unassigned_framing_groups - 1
                
                #sections groups with sufficient available allocation for target_fg (t^ > threshold_score)
                section_groups = normalised_df[target_fg][normalised_df[target_fg] >= threshold_score]
                #critical framing group - lowest sufficient available allocation
                if len(section_groups) <1:
                    req_length = min(required_df[target_fg][dispatch_df[target_fg]>0]*threshold_score)
                    available_length = max(dispatch_df[target_fg][dispatch_df[target_fg] >= 0])
                    raise ValueError(f'Unable to assign framing group {target_fg}: {req_length} required length and {available_length} available length. {unassigned_framing_groups} framing groups unassigned.')
                target_sg = section_groups.idxmin()

                #get final allocation for sg -> fg
                target_allocation = int(required_df.at[target_sg,target_fg]*threshold_score)
                final_dispatch_df.at[target_sg,target_fg] = target_allocation
                assigned_framing_groups.append(target_fg)

                #reduce allocation for target_fg to 0 
                dispatch_df[target_fg] = 0
                #reduce allocation for target_sg to other framing_groups
                dispatch_df.loc[target_sg] = dispatch_df.loc[target_sg]  - target_allocation
                dispatch_df.loc[target_sg] = dispatch_df.loc[target_sg].clip(lower=0)

                #recalc normalised_df
                normalised_df = dispatch_df.div(required_df)


            dispatch_df = final_dispatch_df
                # # Step 4: Identify the index with the smallest value greater than the threshold for each column
                # indices_to_keep = {}
                # for col in sorted_cols:
                #     filtered_values = normalised_df[col][normalised_df[col] > threshold_score]
                #     if not filtered_values.empty:
                #         indices_to_keep[col] = filtered_values.idxmin()  # Get the index with the smallest value above threshold

                # # Step 5: Modify df to retain only identified indices and set other values to 0
                # df_modified = df.copy()
                # for col, idx in indices_to_keep.items():
                #     df_modified.loc[df_modified.index != idx, col] = 0

                # Display the modified DataFrame
            # print(df_modified)

            # #ensure each framing group has only one allocated section_group
            # for target_column in dispatch_df.columns:
            #     # Identify rows where target_column is non-zero
            #     non_zero_target = dispatch_df[target_column] != 0

            #     # Check if there are multiple non-zero values in the target column
            #     if non_zero_target.sum() > 1:
            #         # Iterate through each row that has a non-zero in the target column
            #         for index, row in dispatch_df[non_zero_target].iterrows():
            #             # Check other columns in the row (except the target column)
            #             # for col in dispatch_df.columns:
            #             #     print(col)
            #             #     print(col != target_column)

            #             #delete other values in this row -> each section group can only be used in one framing group
            #             if any(row[col] != 0 for col in dispatch_df.columns if col != target_column):
            #                 # Zero out the target column value in this row
            #                 dispatch_df.at[index, target_column] = 0
            #                 # Break the loop if only one non-zero remains in the target column
            #                 if (dispatch_df[target_column] != 0).sum() == 1:
            #                     break

            # #Keep the first non-zero value per column so only one section group assigned to each framing group
            # non_zero_mask = dispatch_df != 0
            # cumulative_nonzero = non_zero_mask.cumsum()
            # dispatch_df[cumulative_nonzero > 1] = 0


            # #allocate portions (only affects section_groups used across multiple framing groups)
            # required_lengths_if_dispatched = self.section_group_required_length_df.multiply(dispatch_df.div(dispatch_df),fill_value=0)
            # required_lengths_total=required_lengths_if_dispatched.sum(axis=1)
            # required_lengths_total=required_lengths_total.to_frame('total')
            # relative_portion = required_lengths_if_dispatched.div(required_lengths_total['total'],axis=0)
            # relative_portion = relative_portion.fillna(0)
            # dispatch_df = (dispatch_df *relative_portion).round(0)

        self.dispatch_matrix = dispatch_df
        self.normalised_dispatch_matrix = self.dispatch_matrix.div(
            self.section_group_required_length_df
        )


    
    def required_lengths_df(self,return_as:str = 'section_group'):
        #NOTE - this property and next property are not named very clearly
        dispatch_df = self.dispatch_matrix
        required_lengths_if_dispatched = self.section_group_required_length_df.multiply(dispatch_df.div(dispatch_df),fill_value=0)
        if return_as == 'matrix':
            return required_lengths_if_dispatched
        elif return_as == 'section_group':
            return required_lengths_if_dispatched.sum(axis=1)
        elif return_as == 'framing_group':
            return required_lengths_if_dispatched.sum(axis=0)

                
    @property
    def required_lengths_after_dispatch(self):
        req_lengths = self.section_group_required_length_df.copy()
        req_lengths=req_lengths.mask(self.dispatch_matrix == 0, 0)
        return req_lengths.reset_index(names='section_group').to_json(orient='records')
        # return self.section_group_required_length_df.multiply(self.dispatch_matrix.div(self.dispatch_matrix),fill_value=0)
            

    # def get_section_group_max_span(self):
    @property
    def section_group_max_span_df(self) -> pd.DataFrame:
        section_group_df = self.stock.section_group_df
        # section_group_sizegrade = section_group_df[["size", "grade"]]
        section_group_sizegrade = section_group_df[["size", "grade"]]
        section_group_max_span = section_group_sizegrade.copy()       

        for framing_group_name, framing_group in self.structure.framing_groups.items():
            #reset sizegrade in case it was updated with laminations in the previous iteration 
            section_group_sizegrade = section_group_df[["size", "grade"]]

            # if framing_group.name in ["F8", "F9"]:
                # ...


            span_table_df = framing_group.span_table

            # Step 0: update section_group size names based on transverse laminations
            if framing_group.transverse_laminations>1:
                section_group_sizegrade.loc[:,'size'] = str(framing_group.transverse_laminations) + '/' + section_group_sizegrade['size']



            # Step 1: Filter span table in each framing group by sizes and grades in section_group_df
            filtered_df = span_table_df[
                span_table_df[["size", "grade"]]
                .apply(tuple, axis=1)
                .isin(section_group_sizegrade.apply(tuple, axis=1))
            ]

            df_merged = pd.merge(
                section_group_sizegrade,
                filtered_df[["size", "grade", "max_span"]],
                on=["size", "grade"],
                how="left",
            )
            section_group_max_span[framing_group_name] = df_merged["max_span"].values

        return section_group_max_span

    @property
    def section_group_span_capacity_ok_as_boolean(self) -> pd.DataFrame:
        max_span_df = self.section_group_max_span_df
        design_span = self.structure.framing_group_design_spans_df
        cols = self.structure.framing_group_names

        # Stage 1: Perform boolean check where df1 > series
        bool_df = max_span_df[cols].ge(design_span)
        return bool_df

    def section_group_span_capacity_dispatch(self, dispatch_matrix) -> pd.DataFrame:
        
        #get section_group span capacities
        bool_df = self.section_group_span_capacity_ok_as_boolean
        #get framing group names and setup result_df as copy of bool_df
        cols = self.structure.framing_group_names
        result_df = bool_df.astype("object", copy=True)
        
        for col in cols:
            # replace true values (=span OK) with lengths from dispatch_matrix
            result_df.loc[bool_df[col], col] = dispatch_matrix.loc[bool_df[col], col]
            # replace false values with 0
            result_df.loc[~bool_df[col], col] = 0
            # nans are unchanged

        # TODO - replace some 0 values with nans if no data in span table (can't use)?
        # -> nans in self.section_group_max_span_df

        return result_df

    


    def section_group_min_part_size_dispatch(self, dispatch_matrix: pd.DataFrame) -> pd.DataFrame:
        #get framing group minimum part sizes
        min_part_lengths = self.structure.framing_group_min_part_lengths

        for framing_group, min_part_length in min_part_lengths.items():
            #get total length of boards in each section_group that satisfy minimum part length
            ok_min_length = self.stock.section_group_lengths_greater_than(min_part_length)
            ok_min_length = ok_min_length["L_total"]
            #get minimum value from dispatch matrix and ok_min_length available lengths
            min_avail_length = pd.DataFrame({'min_value': ok_min_length.combine(dispatch_matrix[framing_group], min)}) 
            #update available length in dispatch matrix
            dispatch_matrix[framing_group] = min_avail_length
        
        #replace nans (from no lengths available in )
        return dispatch_matrix

    # @property
    # def section_group_spacing_df(self):
    #     framing_cols = self.structure.framing_group_names
    #     df = pd.DataFrame(index=self.stock.section_group_df.index, columns=framing_cols)

    #     for framing_group_name, framing_group in self.structure.framing_groups.items():
    #         if framing_group.design_parameters.SPACING_FROM_DESIGN_PARAMETERS:
    #             df[framing_group_name] = (
    #                 framing_group.design_parameters.spacing_calculation(framing_group.data_dict)
    #             )
    #         else:
    #             for index, row in self.stock.section_group_df.iterrows():
    #                 element_group_id = row.element_group_indices[0]
    #                 element_group = self.stock.element_group_df.loc[element_group_id, :]
    #                 df.loc[index, framing_group_name] = (
    #                     framing_group.design_parameters.spacing_calculation(
    #                         dict(element_group), framing_group.data_dict
    #                     )
    #                 )
    #             # df[framing_group_name] = 'todo'

    #     return df

    @property
    def section_group_quantity_df(self):
        # section_group_spacings = self.section_group_spacing_df
        # df = self.section_group_spacing_df.copy()
        framing_cols = self.structure.framing_group_names
        df = pd.DataFrame(index=self.stock.section_group_df.index, columns=framing_cols)

        for framing_group_name, framing_group in self.structure.framing_groups.items():
            quantities = framing_group.calc_quantities(self.stock)
            df[framing_group_name] = quantities
        return df

    @property
    def member_location_df(self):
        framing_cols = self.structure.framing_group_names
        df = pd.DataFrame(index=self.stock.section_group_df.index, columns=framing_cols)

        for framing_group_name, framing_group in self.structure.framing_groups.items():
            locations = framing_group.calc_locations(self.stock)
            df[framing_group_name] = locations
        return df
    
    @property
    def dispatch_final_assignment(self):
        #check unique assignment
        dispatch_df = self.dispatch_matrix
        
        assignments_per_fg = dispatch_df[dispatch_df > 0].count(axis=0)
        assignment_df = pd.DataFrame(index=dispatch_df.columns)
        assignment_df['section_group']=''
        assignment_df.index.names = ['framing_group']

        if any(assignments_per_fg>1):
            raise ValueError('Error: can not give a final assignment -> non-unique framing group assignment')
        else:
            for ind, row in dispatch_df.iterrows():
                # if len(row[row>0]): #copy over assignment 
                for fg in list(row[row>0].index):
                    assignment_df.at[fg,'section_group']=ind
        return assignment_df

    @property
    def dispatch_results_summary_df(self):
        sg = self.stock.section_group_df
        # framing_cols = self.structure.framing_group_names
        df = pd.DataFrame(index=self.stock.section_group_df.index)
        df["Available Section Group Length"] = sg["L_total"]
        dispatch_df = self.dispatch_matrix
        df["Allocated Section Group Length"] = dispatch_df.sum(axis=1)
        df["Stock Utilisation"] = 0 #


        df["Required Framing Group Length"]  = self.required_lengths_df(return_as='section_group')
        df["threshold_score"] = df["Allocated Section Group Length"]/df["Required Framing Group Length"]
        df["Allocated Framing Groups"] = dispatch_df.ne(0).dot(dispatch_df.columns+',').str[:-1]
        
        
        df.loc['Total'] = df.sum(numeric_only=True, skipna=True)
        df.at['Total','threshold_score']=nan

        df["Stock Utilisation"] = df["Allocated Section Group Length"]/df["Available Section Group Length"] 
        
        return df
        # df["Required Framing Groups Length"]/ 



    @property
    def section_group_required_length_df(self) -> pd.DataFrame:
        df = self.section_group_quantity_df.copy()
        # section_group_spacings = self.section_group_spacing_df
        section_group_quantities = self.section_group_quantity_df
        # df = self.section_group_spacing_df.copy()
        for framing_group_name, framing_group in self.structure.framing_groups.items():
            ...
            df[framing_group_name] = (
                framing_group.length
                * framing_group.transverse_laminations
                * section_group_quantities[framing_group_name]
            )
            # divide framing_group.wdith = framing_group
        return df

    #############################
    # METHOD 2: PART ASSIGNMENT
    #############################
    def create_members(self) -> None:
        member_count = 1
        for framing_group_name, framing_group in self.structure.framing_groups.items():
            framing_group_dispatch = self.dispatch_matrix[framing_group_name]
            for section_group_name, allocated_lengths in framing_group_dispatch.items():
                if allocated_lengths > 0:
                    member_qty = self.section_group_quantity_df.at[section_group_name,framing_group_name]
                    #member_name = f"M{member_count}_{framing_group.name}"
                    member_name = f"M{member_count}"
                    member = Member(name=member_name,qty = member_qty, framing_group_name = framing_group_name) 
                    member.solve_jointing_patterns(
                        framing_group.discrete_joints, 
                        framing_group.span_type,
                        framing_group.max_parts_per_length,
                        framing_group.transverse_laminations)
                    self.structure.add_member(member_name, member)
                    self.structure.add_parts(section_group_name, member)
                    member_count = member_count + 1

    def create_element_group_cutting_patterns(self, pattern_type: str = 'single_part_only'):
        section_groups = self.stock.section_group_df.index.tolist()
        unique_parts_per_section_group = {key: pd.Series(dtype=object) for key in section_groups}

        for member_name, member in self.structure.members.items():         
            # set of parts chopped from inventory boards to all potential part lengths
            #unique_part_lengths = member.jointing_patterns.unique_lengths
            unique_parts = member.jointing_patterns.unique_parts
            if (self.dispatch_matrix[member.framing_group_name]>0).sum()>1:
                raise ValueError('Error - multiple section_groups allocated to framing groups')
            
            section_group_name = self.dispatch_matrix[member.framing_group_name][self.dispatch_matrix[member.framing_group_name]>0].index[0]
            parts_used_in_member_jointing_patterns = member.jointing_patterns.unique_part_names_used_in_jointing_patterns 
            # aggregate all member part lengths into each section group, before making element_group_cutting_patterns
            unique_parts_per_section_group[section_group_name] = pd.concat([
                unique_parts_per_section_group[section_group_name],
                unique_parts[parts_used_in_member_jointing_patterns]
            ])
            unique_parts_per_section_group[section_group_name].drop_duplicates(inplace=True)

        # add element group cutting patterns for each section group
        for section_group_name, unique_parts in unique_parts_per_section_group.items():
            if len(unique_parts)>0:
                self.add_element_group_cutting_patterns(section_group_name, unique_parts, pattern_type)
            
    def add_element_group_cutting_patterns(self, section_group_name:str, unique_parts:pd.Series, pattern_type: str)->None:
        if self.cutting_patterns is None:
            self.cutting_patterns = {}

        element_group_df = (self.stock.element_group_df[self.stock.element_group_df['section_group']==section_group_name]).copy()
        cutting_patterns = ElementCuttingPatterns(element_group_df, unique_parts, pattern_type= pattern_type)
        cutting_patterns.rename_patterns()
        self.cutting_patterns[section_group_name] = cutting_patterns
        



    # def create_assembly_patterns(self):
    #     # Initialize the given dataframes
    #     if self.assembly_patterns is None:
    #         self.assembly_patterns = {}
    #     for member_name in self.structure.member_names:
    #         assembly_patterns = AssemblyPatterns(
    #             jointing_patterns=self.structure.members[member_name].jointing_patterns,
    #             cutting_patterns=self.cutting_patterns[member_name]
    #         )
    #         assembly_patterns.rename_patterns(member_name)
    #         self.assembly_patterns[member_name] = assembly_patterns

    def export_dispatch(self, filepath):
        export_me = {
            "element_groups": self.stock.element_group_df,
            "section_groups": self.stock.section_group_df,
            "framing_groups": self.structure.framing_group_df,
            "0A_quantity_matrix": self.section_group_quantity_df,
            "0B_required_length_matrix": self.section_group_required_length_df,
            "1A_span_capacity_matrix": self.section_group_max_span_df,
            "1B_span_capacity_ok_matrix":self.section_group_span_capacity_ok_as_boolean,
            #2 - framing group min part length included in framing groups
            "3A_dispatch_matrix": self.dispatch_matrix,
            "3B_normalised_dispatch_matrix": self.normalised_dispatch_matrix,
            "3C_dispatch_results_summary": self.dispatch_results_summary_df,
            "3D_dispatch_final_assigment": self.dispatch_final_assignment
        }

        for export_name, df in export_me.items():
            df.to_csv(f"{filepath}//{export_name}.csv")


    def export_matrices(self, filepath):
        export_me = {
            "J_jointing_patterns": self.J_matrix,
            "C_cutting_patterns": self.C_matrix,
            "E_element_group_demand": self.E_matrix ,
            "Q_e_element_group_quantities": self.Q_e,
            "M_member_supply": self.M_matrix,
            "Q_m_member_quantities": self.Q_m,
            "w_waste_vector": self.w_vector,
            "l_part_length_vector": self.l_vector
        }

        for export_name, df in export_me.items():
            df.to_csv(f"{filepath}//{export_name}.csv")

    def export_parts(self,filepath: Optional[str]=None):
        ...
        if filepath is not None:
            self.structure.parts.to_csv(f"{filepath}//parts.csv")
        parts_3d = self.structure.parts.copy()
        parts_3d['D'] = 0
        parts_3d['B'] = 0
        for ind, row in parts_3d.iterrows():
            section_group = self.stock.section_group_df.loc[row['section_group']]
            element_group_id = section_group.element_group_indices[0]
            element_group = self.stock.element_group_df.loc[element_group_id]
            parts_3d.at[ind,'D'] = element_group.D
            parts_3d.at[ind,'B'] = element_group.B
        parts_3d.rename(columns={"part_length":"L"}, inplace=True)
        parts_3d['part_name'] = parts_3d.index
        if filepath is not None:
            parts_3d.to_csv(f"{filepath}//parts_3d.csv")
            parts_3d.to_json(f"{filepath}//parts_3d.json", orient="records")
        else:
            return parts_3d.to_json(orient="records")

    def compile_elements(self) -> pd.DataFrame:
        elements = []
        for ind, row in self.stock.element_group_df.iterrows():
            for element_num in range(row.qty):
                element_dict = {'element_id': f'{ind}-{element_num+1}','D':row.D, 'B':row.B, 'L':row.L, 'element_group_id': ind}
                elements.append(element_dict)
        element_df = pd.DataFrame(elements)
        return element_df

    def export_elements(self,filepath: Optional[str]=None):
        element_df = self.compile_elements()
        if filepath is not None:
            element_df.to_json(f"{filepath}//element_3d.json", orient="records")
        else:
            return element_df.to_json(orient="records")


    def export_element_groups(self,filepath: Optional[str]=None):
        # ...
        element_group_df = self.stock.element_group_df.copy()
        element_group_df['element_group_id']=element_group_df.index
        if filepath is not None:
            element_group_df.to_csv(f"{filepath}//element_groups.csv")
            element_group_df.to_json(f"{filepath}//element_groups.json", orient="records")
        else:
            return element_group_df.to_json(orient="records")

    def export_section_groups(self,filepath: Optional[str]=None):
        section_group_3d = self.stock.element_group_df[['D', 'B', 'L','qty']].copy()
        section_group_3d['section_group_id'] = section_group_3d.index
        section_group_3d['section_group_id_and_qty'] = section_group_3d['section_group_id']  + ' (x' + section_group_3d['qty'].astype(str) + ')'
        if filepath is not None:
            section_group_3d.to_csv(f"{filepath}//section_groups.csv")
            section_group_3d.to_json(f"{filepath}//section_groups_3d.json", orient="records")
        else:
            return section_group_3d.to_json(orient="records")

    def compile_member_groups(self) -> pd.DataFrame:
        member_dicts = []
        
        for member_name, member in self.structure.members.items():
            #NOTE - should make this function a property or method
            # section_group_name = self.dispatch_matrix[member.framing_group_name][self.dispatch_matrix[member.framing_group_name]>0].index.values[0]
            # self.section_group_quantity_df.at[section_group_name,framing_group_name]
            # member_spacing = self.section_group_spacing_df.at[section_group_name,member.framing_group_name]
            member_dict = {"member_group_id": member_name, "qty": member.qty, "L":member.jointing_patterns.support.target_length, 'framing_group_name':member.framing_group_name}#, "spacing": member_spacing}
            member_dicts.append(member_dict)
        member_group_df = pd.DataFrame(member_dicts)
        member_group_df.set_index('member_group_id', inplace=True)
        member_group_df['member_group_id']=member_group_df.index
        member_group_df['member_group_and_qty'] = member_group_df['member_group_id']  + ' (x' + member_group_df['qty'].astype(str) + ')'
        
        return member_group_df

    def compile_members(self) -> pd.DataFrame:
        member_group_df = self.compile_member_groups()
        members = []
        for ind, row in member_group_df.iterrows():
            for member_num in range(row.qty):
                member_dict = {'member_id': f'{ind}-{member_num+1}','L':row.L, 'member_group_id': ind}
                members.append(member_dict)
        members = pd.DataFrame(members)
        return members

    def export_member_groups(self,filepath: Optional[str]=None):
        member_group_df = self.compile_member_groups()
        if filepath is not None:
            member_group_df.to_json(f"{filepath}//member_groups.json", orient="records")            
            member_group_df.to_csv(f"{filepath}//member_groups.csv")
        else:
            return member_group_df.to_json(orient="records")

    def export_members(self,filepath: Optional[str]=None):
        member_df = self.compile_members()
        if filepath is not None:
            member_df.to_json(f"{filepath}//members_1d.json", orient="records")
        else:
            return member_df.to_json(orient="records")


    def all_cutting_patterns(self) -> pd.DataFrame:
        #TODO - move this elsewhere
        all_cutting_patterns = pd.DataFrame(columns=['element_group_id', 'part_id'])
        for section_group_name, _ in self.cutting_patterns.items():
            cur_cp = self.cutting_patterns[section_group_name].df[['element_group_id', 'part_id']].copy()
            cur_cp['cutting_pattern_id'] = cur_cp.index
            cur_cp['part_offset'] = cur_cp['part_id']
            cur_cp['total_part_length']=0
            #cur_cp.at[:,'part_offset'] = [0]
            for ind,row in cur_cp.iterrows():
                last_offset = 0
                part_offsets = [last_offset]
                #
                for part in row.part_id:
                    last_offset = last_offset + self.structure.parts.loc[part].part_length
                    part_offsets.append(last_offset)
                #total part length = last part offset
                cur_cp.at[ind,'total_part_length']=part_offsets[-1:]
                #drop last offset
                part_offsets = part_offsets[:-1]
                cur_cp.at[ind,'part_offset']=part_offsets

                # cur_cp['total_part_length'].apply(lambda x: x[-1])
            all_cutting_patterns = pd.concat([all_cutting_patterns, cur_cp])
            
            # all_cutting_patterns['part_offset'].apply(lambda x: x[-1])
        all_cutting_patterns.rename(columns={"element_group_id": "element_group_id"}, inplace=True)
        return all_cutting_patterns


    def export_cutting_patterns(self, filepath: Optional[str]=None):
        #first_member = self.structure.member_names[0]
        all_cutting_patterns = self.all_cutting_patterns()
        if filepath is not None:
            all_cutting_patterns.to_csv(f"{filepath}//cutting_patterns.csv")
            all_cutting_patterns.to_json(f"{filepath}//cutting_patterns.json", orient="records")
        else:
            return all_cutting_patterns.to_json(orient="records")

    def all_jointing_patterns(self)->pd.DataFrame:
        #first_member = self.structure.member_names[0]
        all_jointing_patterns = []
        for member in self.structure.member_names:
            cur_jps = self.jointing_patterns[member].part_count_by_pos
            for ind, row in cur_jps.iterrows():
                part_ids = list(row.values)
                part_ids = [v for v in part_ids if v != '-']
                part_offsets = [0]
                last_offset = 0
                for part in part_ids:
                    last_offset = last_offset + self.structure.parts.loc[part].part_length
                    part_offsets.append(last_offset)
                part_offsets = part_offsets[:-1]

                #repeat parts for transversely laminated members
                num_parts = len(part_ids)
                part_multiplier = self.jointing_patterns[member].part_multiplier
                part_ids = part_ids * part_multiplier
                part_offsets = part_offsets*part_multiplier
                transverse_offset = list(range(part_multiplier))*num_parts
                transverse_offset.sort()
                jp_dict = {
                    'jointing_pattern_id':ind,
                    'member_group_id': member,
                    'part_id': part_ids,
                    'part_offset': part_offsets,
                    'transverse_offset': transverse_offset
                }
                all_jointing_patterns.append(jp_dict)
        all_jointing_patterns=pd.DataFrame(all_jointing_patterns)
        return all_jointing_patterns

    def export_jointing_patterns(self, filepath: Optional[str]=None):
        all_jointing_patterns=self.all_jointing_patterns()
        
        if filepath is not None:
            all_jointing_patterns.to_json(f"{filepath}//jointing_patterns.json", orient="records")
            all_jointing_patterns.set_index('jointing_pattern_id', inplace=True)
            all_jointing_patterns.to_csv(f"{filepath}//jointing_patterns.csv")
        else:
            return all_jointing_patterns.to_json(orient="records")


    @staticmethod
    def ilp_result_to_df(result:dict)->pd.DataFrame:
        #strip 'x_' or 'y_' from design variables
        result = {k[2:]: v for k, v in result.items()}
        #remove 0 values
        result = {k: v for k, v in result.items() if v != 0}
        #conver to dataframe
        result_df = pd.DataFrame.from_dict(result, orient='index',columns=['freq'])
        return result_df


    def compile_jointing_results(self, result_x) ->pd.DataFrame:
        all_jointing_patterns = self.all_jointing_patterns()
        result_x_df = self.ilp_result_to_df(result_x)
        result_x_df.reset_index(names='jointing_pattern_id', inplace=True)
        result_x_df = pd.merge(result_x_df, all_jointing_patterns[['jointing_pattern_id', 'member_group_id']], on='jointing_pattern_id')
        result_x_df['remaining']=result_x_df['freq'].values

        # all_jps = []
        member_to_jp_df = self.compile_members()
        # member_df=member_df.copy()
        member_to_jp_df.drop('L', axis=1, inplace=True)
        member_to_jp_df['jointing_pattern_id']=''
        if self.MEMBER_MULTIPLIER > 1:
            member_to_jp_df['repeat_structure']=0
            repeat_member_df = pd.DataFrame(columns=member_to_jp_df.columns)
            for repeat_struct in range(self.MEMBER_MULTIPLIER):
                member_to_jp_df['repeat_structure'] = repeat_struct+1
                repeat_member_df=pd.concat([repeat_member_df,member_to_jp_df],ignore_index=True)
            #rename members with structure count so member_id can still be used as index
            # repeat_member_df['member_id'] + 'x'+repeat_member_df['repeat_structure'].apply(str)
            member_to_jp_df = repeat_member_df


        for ind, row in member_to_jp_df.iterrows():

            matching_jps = result_x_df[result_x_df['member_group_id']==row['member_group_id']]
            if len(matching_jps[matching_jps['remaining']>0])>0:
                #find and append first assigned pattern
                first_match_ind = matching_jps[matching_jps['remaining']>0].index[0]
                matching_cp = result_x_df.loc[first_match_ind,'jointing_pattern_id']
                member_to_jp_df.at[ind,'jointing_pattern_id'] = matching_cp
                #reduce remaining availability for cutting pattern
                result_x_df.at[first_match_ind,'remaining']=result_x_df.at[first_match_ind,'remaining']-1
            else:
                #no jointing patterns assigned to remaining member - append blank member
                #NOTE - this it violates ILP constraint so only happens if exporting with no solution
                member_to_jp_df.at[ind,'jointing_pattern_id'] = None #all_jps.append(ind)

        soln = member_to_jp_df
        #add parts
        soln = soln.merge(all_jointing_patterns[['jointing_pattern_id','part_id']],how='left')
        soln["part_count"] = soln["part_id"].apply(len)
        return soln

    def export_jointing_results(self,result_x, filepath: Optional[str]=None):
        soln = self.compile_jointing_results(result_x)
        if filepath is not None:
            soln.to_json(f"{filepath}//solution_x_jointing_patterns.json", orient="records")
            soln.to_csv(f"{filepath}//solution_x_jointing_patterns.csv", index=False)
        else:
            return soln.to_json(orient="records")

    def export_jointing_results_raw(self, result_x, filepath: Optional[str]=None):
        soln = pd.DataFrame.from_dict(result_x.items())
        soln.set_index(0)

        if filepath is not None:
            # soln.to_json(f"{filepath}//solution_y_raw_cutting_patterns.json", orient="records")
            soln.to_csv(f"{filepath}//solution_x_raw_jointing_patterns.csv")
        else:
            return soln.to_json(orient="records")


    def export_cutting_results_raw(self, result_y, filepath: Optional[str]=None):
        soln = pd.DataFrame.from_dict(result_y.items())
        soln.set_index(0)

        if filepath is not None:
            # soln.to_json(f"{filepath}//solution_y_raw_cutting_patterns.json", orient="records")
            soln.to_csv(f"{filepath}//solution_y_raw_cutting_patterns.csv")
        else:
            return soln.to_json(orient="records")

    def compile_cutting_results(self,result_y) -> pd.DataFrame:
        all_cutting_patterns = self.all_cutting_patterns()
        result_y_df = self.ilp_result_to_df(result_y)
        result_y_df.reset_index(names='cutting_pattern_id', inplace=True)
        result_y_df = pd.merge(result_y_df, all_cutting_patterns[['cutting_pattern_id', 'element_group_id']], on='cutting_pattern_id')


        result_y_df['remaining']=result_y_df['freq'].values
        element_to_cp_df = self.compile_elements()
        
        element_to_cp_df = element_to_cp_df[['element_id', 'element_group_id']]#.drop('L', axis=1, inplace=True)
        element_to_cp_df['cutting_pattern_id']=''
        for ind, row in element_to_cp_df.iterrows():
            matching_cps = result_y_df[result_y_df['element_group_id']==row['element_group_id']]
            if len(matching_cps[matching_cps['remaining']>0])>0:
                #find and append first assigned pattern
                first_match_ind = matching_cps[matching_cps['remaining']>0].index[0]
                matching_cp = result_y_df.loc[first_match_ind,'cutting_pattern_id']
                element_to_cp_df.at[ind,'cutting_pattern_id'] = matching_cp
                #reduce remaining availability for cutting pattern
                result_y_df.at[first_match_ind,'remaining']=result_y_df.at[first_match_ind,'remaining']-1
            else:
                #no cutting patterns assigned to remaining element_groups - append blank element_group
                # all_cps.append(ind)
                element_to_cp_df.at[ind,'cutting_pattern_id'] = None
        
        soln = element_to_cp_df #pd.DataFrame(all_cps, columns=['y_c'])

        #add parts produced per element
        soln = soln.merge(all_cutting_patterns[['cutting_pattern_id','part_id']],how='left')
        soln["part_count"] = soln["part_id"].apply(lambda x: len(x) if type(x) == list else x)

        #add total part length per element and cutting wastage
        soln = soln.merge(all_cutting_patterns[['cutting_pattern_id','total_part_length']],how='left')
        element_group_df = self.stock.element_group_df.reset_index(names='element_group_id')
        soln = soln.merge(element_group_df[['element_group_id','L']],how='left')
        soln['T_w'] = soln['L']-soln['total_part_length']

        ...
        return soln


    def compile_part_results(self,result_x, result_y)->pd.DataFrame:
        jointing_results = self.compile_jointing_results(result_x)
        cutting_results = self.compile_cutting_results(result_y)
        member_parts = jointing_results['part_id'].explode().to_list()
        cutting_results['part_id']=cutting_results['part_id'].fillna('None')
        parts = self.structure.parts

        part_results = deepcopy(cutting_results[['element_id','element_group_id']])
        unused_parts = []
        unused_part_lengths = []
        for _,row in cutting_results.iterrows():
            cur_unused_parts = []
            cur_unused_part_length = []
            if row['part_id']!='None':
                for part_id in row['part_id']:
                    if part_id in member_parts:
                        #part is used in a member - remove and move on
                        member_parts.remove(part_id)
                        #TODO - add matching member
                    else:
                        #part is not used in a member - add to unused part list
                        cur_unused_parts.append(part_id)
                        cur_unused_part_length.append(parts.loc[part_id].part_length)
            else:
                ...
            unused_parts.append(cur_unused_parts)
            unused_part_lengths.append(cur_unused_part_length)
        part_results['unused_parts']=unused_parts
        part_results['unused_part_lengths']=unused_part_lengths
        part_results["unused_part_count"] = part_results["unused_parts"].apply(lambda x: len(x) if type(x) == list else x)
        part_results["total_unused_part_lengths"] = part_results["unused_part_lengths"].apply(lambda x: sum(x) if type(x) == list else x)
        return part_results

    def export_part_results(self,result_x, result_y, filepath: Optional[str]=None):
        soln = self.compile_part_results(result_x, result_y)
        soln.reset_index(inplace=True)
        if filepath is not None:
            soln.to_json(f"{filepath}//results_summary_parts.json", orient="records")
            soln.to_csv(f"{filepath}//results_summary_parts.csv",index=False)
        else:
            return soln.to_json(orient="records")           


    def export_wastage_results(self,result_x, result_y, filepath: Optional[str]=None):
        soln = self.w_i(result_x, result_y)
        soln.reset_index(inplace=True)
        if filepath is not None:
            soln.to_json(f"{filepath}//results_summary_wastage.json", orient="records")
            soln.to_csv(f"{filepath}//results_summary_wastage.csv",index=False)
        else:
            return soln.to_json(orient="records")           


    def export_cutting_results(self, result_y, filepath: Optional[str]=None):
        soln = self.compile_cutting_results(result_y)
        if filepath is not None:
            soln.to_json(f"{filepath}//solution_y_cutting_patterns.json", orient="records")
            soln.to_csv(f"{filepath}//solution_y_cutting_patterns.csv",index=False)
        else:
            return soln.to_json(orient="records")
        
    def compile_member_results(self,result_x) -> pd.DataFrame:
        member_df = pd.DataFrame(index=self.n_Jm.index, columns = ['n_Pm', 'n_Jm', 'q_pm', 'q_pm^req'])
        member_df['n_Pm'] = self.n_Pm
        member_df['n_Jm'] = self.n_Jm
        member_df['q_pm'] = self.q_pm(result_x)
        member_df['q_pm^req'] = self.compile_jointing_results(result_x).groupby('member_group_id')['part_count'].sum(list)
        return member_df
    
    def export_member_results_summary(self, result_x, filepath:Optional[str]=None):
        soln = self.compile_member_results(result_x)
        soln.reset_index(inplace=True)
        if filepath is not None:
            soln.to_json(f"{filepath}//results_summary_members.json", orient="records")
            soln.to_csv(f"{filepath}//results_summary_members.csv",index=False)
        else:
            return soln.to_json(orient="records")
        

        
    def compile_section_group_results(self,result_y) -> pd.DataFrame:
        df = pd.DataFrame(index=self.n_Pk.index, columns = ['n_Pk', 'n_Ck', 'q_k^cut'])
        df['n_Pk'] = self.n_Pk
        # mediator.n_Pk
        # member_df['n_Jm'] = self.n_Jm
        # df['q_pm'] = self.q_pm(result_y)
        df['q_k^cut'] = self.compile_cutting_results(result_y).groupby('section_group_id')['part_count'].sum(list)
        
        return df
    
    
    def compile_element_group_results(self,result_y) -> pd.DataFrame:
        element_group_df = self.stock.element_group_df
        df = pd.DataFrame(index=element_group_df.index, columns = ['n_Pi', 'n_Ci', 'q_i^cut','t_i','t_i^cut','t_i^cut/t_i'])
        #remove total column
        df['n_Pi'] = self.n_Pi
        # mediator.n_Pk
        # member_df['n_Jm'] = self.n_Jm
        # df['q_pm'] = self.q_pm(result_y)

        result_df = self.compile_cutting_results(result_y)
        #cps = self.all_cutting_patterns()
        #numper of parts made: (NOT equal to number of parts possible)
        # result_df.groupby('element_group_id')['part_count'].count()
        df['n_Ci'] = self.n_Ci['Total'] #cps.groupby('element_group_id')['cutting_pattern_id'].count()
        df['q_i^cut'] = result_df.groupby('element_group_id')['part_count'].count()
        df['t_i'] = element_group_df['L_total']
        df['t_i^cut'] = df['q_i^cut']*element_group_df['L']

        df['t_i^cut/t_i'] = df['t_i^cut']/df['t_i']
        #result_df.groupby('total_part_length')['part_count'].sum(list)
        
        return df
    
    
    def export_section_group_results_summary(self, result_y, filepath:Optional[str]=None):
        soln = self.compile_section_group_results(result_y)
        soln.reset_index(inplace=True)
        if filepath is not None:
            soln.to_json(f"{filepath}//results_summary_section_groups.json", orient="records")
            soln.to_csv(f"{filepath}//results_summary_section_groups.csv",index=False)
        else:
            return soln.to_json(orient="records")
        
    def export_element_group_results_summary(self, result_y, filepath:Optional[str]=None):
        soln = self.compile_element_group_results(result_y)
        soln.reset_index(inplace=True)
        if filepath is not None:
            soln.to_json(f"{filepath}//results_summary_element_groups.json", orient="records")
            soln.to_csv(f"{filepath}//results_summary_element_groups.csv",index=False)
        else:
            return soln.to_json(orient="records")
        



    @property
    def jointing_patterns(self) -> dict[str,MemberJointingPatterns]:
        #add as a property to access jointing patterns in same way as other patterns
        new_dict = {}
        for member_name in self.structure.member_names:
            new_dict[member_name]  = self.structure.members[member_name].jointing_patterns
        return new_dict

    @property
    def all_assembly_pattern_names(self) -> list[str]:
        names = []
        for member_name in self.structure.member_names:
            names = names + list(self.assembly_patterns[member_name].df.index)
        return names
    
    @property
    def all_cutting_pattern_names(self) -> list[str]:
        names = []
        for section_group_name, _ in self.cutting_patterns.items():
            names = names + list(self.cutting_patterns[section_group_name].df.index)
        return names

    @property
    def all_jointing_pattern_names(self) -> list[str]:
        names = []
        for member_name in self.structure.member_names:
            names = names + list(self.jointing_patterns[member_name].df.index)
        return names    


    @property
    def n_Ci(self)->pd.DataFrame:
        n_Ci_df = pd.DataFrame(index=self.stock.element_group_names)

        for section_group_name, _ in self.cutting_patterns.items(): 
            cps = self.cutting_patterns[section_group_name].df

            n_Ci_df[section_group_name]=''
            for element_group_name in self.stock.element_group_names:
                n_Ci_m = len(cps[cps['element_group_id']==element_group_name])
                n_Ci_df.at[element_group_name,section_group_name]=n_Ci_m

        n_Ci_df['Total']=n_Ci_df.sum(axis=1)
        n_Ci_df.loc['Total']= n_Ci_df.sum()
        return n_Ci_df

    @property
    def n_Pi(self)->pd.DataFrame:
        ...
        df = pd.DataFrame(index=self.stock.element_group_df.index,columns=['n_Pi'])
        df['n_Pi']=0
        for section_group_name, _ in self.cutting_patterns.items(): 
            cps = self.cutting_patterns[section_group_name].df
            new_part_count = cps.groupby('element_group_id')['part_id'].count()
            df.loc[new_part_count.index,'n_Pi']=df['n_Pi'].loc[new_part_count.index]+new_part_count
        return df
    
    @property
    def n_Pm(self)->pd.DataFrame:
        #NOTE - this code returns utilised parts only (removes parts e.g. which are not found in the jointing patterns)
        # member_names = self.structure.member_names
        # n_Pm_df = pd.DataFrame(index=member_names,columns=['n_Pm'])

        # for member_name, member in self.structure.members.items():    
        #     n_Pm_df.at[member_name,'n_Pm'] = member.jointing_patterns.n_Pm
        # return n_Pm_df
    
        #NOTE - this code returns all possible parts 
        # First DataFrame (mapping of framing_group to section_group)
        df1 = self.dispatch_final_assignment
        # Second DataFrame (list of parts)
        df2 = self.structure.parts
        df2 = df2.reset_index(names='part_id')

        df3 = self.compile_member_groups()
        # Reset index on df1 to make framing_group a column for merging
        df1_reset = df1.reset_index()
        # Group df1 by section_group to aggregate framing_group values as a list
        df1_grouped = df1_reset.groupby('section_group')['framing_group'].agg(list).reset_index()
        # Merge df2 with the grouped df1 to add framing_group column based on section_group
        merged_df = df2.merge(df1_grouped, on='section_group', how='left')
        # Merge df3 onto merged_df by matching `framing_group_name` with `framing_group`
        df3_reset = df3.set_index('member_group_id').reset_index()
        final_merged_df = merged_df.explode('framing_group').merge(
            df3_reset[['framing_group_name', 'member_group_id']],
            left_on='framing_group',
            right_on='framing_group_name',
            how='left'
        ).drop(columns=['framing_group_name'])
        parts_per_member_group = final_merged_df.groupby('member_group_id').size().reset_index(name='num_parts')
        parts_per_member_group.set_index('member_group_id',inplace=True)
        return parts_per_member_group

    @property
    def n_Jm(self)->pd.DataFrame:
        member_names = self.structure.member_names
        n_Jm_df = pd.DataFrame(index=member_names,columns=['n_Jm'])

        for member_name, member in self.structure.members.items():    
            n_Jm_df.at[member_name,'n_Jm'] = member.jointing_patterns.n_Jm
        return n_Jm_df
    


    @property
    def n_Pk(self)->pd.DataFrame:
        structure_parts = self.structure.parts
        section_group_names= self.stock.section_group_names
        n_Pk_df = pd.DataFrame(index=section_group_names,columns=['n_Pk'])
        for section_group_name, _ in self.cutting_patterns.items():
            n_Pk = len(structure_parts[structure_parts['section_group']==section_group_name])
            n_Pk_df.at[section_group_name,'n_Pk'] = n_Pk
        return n_Pk_df



    def q_pm(self, result_x)->pd.DataFrame:
        '''quantity of parts produced per member group, from optimisation solution'''
        x_results = self.compile_jointing_results(result_x)
        x_results.set_index('member_group_id')
        df = x_results.groupby("member_group_id")["part_count"].sum().reset_index()
        df.set_index('member_group_id', inplace=True)
        return df

    def q_pi(self, result_y)->pd.DataFrame:
        '''quantity of parts produced per element group, from optimisation solution'''
        y_results = self.compile_cutting_results(result_y)
        y_results.set_index('element_group_id')
        df = y_results.groupby("element_group_id")["part_count"].sum().reset_index()
        df.set_index('element_group_id',inplace=True)
        return df

    def w_i(self, result_x, result_y)->pd.DataFrame:
        '''waste produced per element group, from optimisation solution'''
        y_results = self.compile_cutting_results(result_y)
        
        part_results = self.compile_part_results(result_x,result_y)
        merge_df = y_results.merge(part_results[['element_id','unused_part_count','total_unused_part_lengths']],how='left')
        merge_df.set_index('element_group_id')
        ...
        soln = merge_df.groupby("element_group_id")[["part_count", "unused_part_count", "total_unused_part_lengths", "T_w"]].sum()#.reset_index()
        soln.rename(columns={"total_unused_part_lengths": "T_p"},inplace=True)
        soln['f']=soln['T_w']+soln['T_p']+soln['part_count']-soln['unused_part_count']
        soln.rename({'part_count':'q^cut'},inplace=True)
        soln = soln.reindex(self.stock.element_group_df.index)

        element_group_results = self.compile_element_group_results(result_y)
        soln['f_hat'] = 1-(soln['T_p']+soln['T_w'])/element_group_results['t_i^cut']
        return soln



    @property
    def C_cp(self):
        ...
        df = pd.DataFrame(index=self.all_cutting_pattern_names, columns=self.structure.part_names)
        df.fillna(0, inplace=True)
        for section_group_name, _ in self.cutting_patterns.items():
            #get member assembly patterns
            C_cp_k = self.cutting_patterns[section_group_name].part_count_df
            df = df.add(C_cp_k, fill_value=0)
        return df

    @property
    def C_matrix(self):
        return self.C_cp

    @property
    def C_ce(self):
        ...
        df = pd.DataFrame(index=self.all_cutting_pattern_names, columns=self.stock.element_group_names)
        df.fillna(0, inplace=True)
        for section_group_name, _ in self.cutting_patterns.items():
            #get member assembly patterns
            C_ce_k = self.cutting_patterns[section_group_name].element_count_df
            # C_ce_k = self.cutting_patterns[member_name].element_group_count_df
            df = df.add(C_ce_k, fill_value=0)
        return df
    
    @property
    def E_matrix(self):
        return self.C_ce

    @property
    def C_cw(self):
        ...
        df = pd.DataFrame(index=self.all_cutting_pattern_names, columns=['waste'])
        df.fillna(0, inplace=True)
        for section_group_name, _ in self.cutting_patterns.items():
            #get member assembly patterns
            C_cw_k = self.cutting_patterns[section_group_name].waste_df
            df = df.add(C_cw_k, fill_value=0)
        return df

    @property
    def w_vector(self):
        return self.C_cw

    @property
    def l_vector(self):
        #part length vector
        return self.structure.parts[['part_length']]

    @property
    def J_jp(self):
        ...
        df = pd.DataFrame(index=self.all_jointing_pattern_names, columns=self.structure.part_names)
        df.fillna(0, inplace=True)
        for member_name in self.structure.member_names:
            #get member assembly patterns
            J_jp_k = self.jointing_patterns[member_name].part_count_df
            df = df.add(J_jp_k, fill_value=0)
        return df

    @property
    def J_matrix(self):
        return self.J_jp

    @property
    def J_jm(self):
        ...
        df = pd.DataFrame(index=self.all_jointing_pattern_names, columns=self.structure.member_names)
        df.fillna(0, inplace=True)
        for member_name in self.structure.member_names:
            #get member assembly patterns
            J_jm_k = self.jointing_patterns[member_name].members_per_joint_pattern
            df = df.add(J_jm_k, fill_value=0)
        return df

    @property
    def M_matrix(self):
        return self.J_jm


    @property
    def Q_m(self):
        return self.structure.member_quantities

    @property
    def Q_e(self):
        return self.stock.element_group_quantities

