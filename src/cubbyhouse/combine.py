"""TODO"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
from cubbyhouse.array import MemberJointingPatterns
from itertools import combinations_with_replacement

#previously BoardPartSet
@dataclass
class ElementCuttingPatterns:
    """TODO"""

    inventory: pd.DataFrame  # boards
    #possible_lengths: list[float]
    unique_parts: Optional[pd.Series] = None
    pattern_type: str = 'single_part_only'
    df: pd.DataFrame = field(init=False, repr=False)
    #part_groups: pd.DataFrame = field(init=False, repr=False)
    #cutting_patterns: pd.DataFrame = field(init=False, repr=False)


    part_count_df: pd.DataFrame = field(init=False, repr=False)
    element_count_df: pd.DataFrame = field(init=False, repr=False)
    waste_df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.pattern_type == 'single_part_only':
            self.generate_patterns()
            self.generate_matrices()
        elif self.pattern_type == 'no_usable_residual':
            self.generate_patterns_no_usable_residual()
            self.generate_matrices_no_usable_residual()
        else:
            raise ValueError(f'pruning type {self.pruning_type} not recognised')
        #self.generate_part_groups()
        

    def generate_patterns(self):
        """TODO"""
        # df with all boards
        all_board_df = pd.DataFrame(columns=["element_group_id", "board_length"])
        all_board_df["element_group_id"] = self.inventory.index.values
        all_board_df["board_length"] = self.inventory["L"].values
        all_board_df["key"] = 1

        # df with all lengths
        length_df = pd.DataFrame(columns=["part_length"])
        length_df["part_length"] = self.possible_lengths
        # length_df["pos_length"] = self.possible_lengths
        length_df["key"] = 1

        # merge
        all_boards_any_length = all_board_df.merge(length_df, on="key").drop(
            "key", axis=1
        )

        # calculate utilisation
        all_boards_any_length["util"] = (
            all_boards_any_length["part_length"] / all_boards_any_length["board_length"]
        )
        all_boards_any_length["util"] = all_boards_any_length["util"].round(4)

        all_boards_any_length["waste"] = all_boards_any_length["board_length"] - all_boards_any_length["part_length"]

        # remove any boards in a position with negative wastage
        # (those not long enough to go between two points / fit required length)
        all_boards_any_length = all_boards_any_length[
            all_boards_any_length["util"] <= 1
        ]
        all_boards_any_length.reset_index(inplace=True, drop=True)

        #
        part_length_to_id = {value: key for key, value in self.unique_parts.items()}
        all_boards_any_length['part_id'] = all_boards_any_length['part_length'].map(part_length_to_id)
        all_boards_any_length = all_boards_any_length[['element_group_id', 'part_id', 'board_length', 'part_length', 'util', 'waste']]

        cutting_pattern_names = [f'C{i}' for i in range(len(all_boards_any_length))]
        all_boards_any_length.index = cutting_pattern_names


        self.df = all_boards_any_length

    def generate_patterns_no_usable_residual(self):
        board_df = pd.DataFrame(columns=["element_group_id", "board_length"])
        board_df["element_group_id"] = self.inventory.index.values
        board_df["board_length"] = self.inventory["L"].values
        part_lengths = self.possible_lengths
        min_part_length = min(part_lengths)
        part_length_to_id = {value: key for key, value in self.unique_parts.items()}

        result = []

        for _, board in board_df.iterrows():
            board_length = board['board_length']
            
            valid_combinations = []
            max_parts = int(board_length // min(part_lengths))
            for r in range(1, max_parts + 1):
                for combo in combinations_with_replacement(part_lengths, r):
                    if sum(combo) <= board_length and (board_length - sum(combo)) < min_part_length:
                        valid_combinations.append(combo)
            
            for combo in valid_combinations:
                result.append({
                    'element_group_id': board['element_group_id'],                    
                    'part_id': [part_length_to_id[v] for v in combo],
                    'board_length': board_length,
                    'part_combination': combo,
                    'sum_part_length': sum(combo),
                    'waste': board_length - sum(combo)
                })
        result_df = pd.DataFrame(result)
        
        cutting_pattern_names = [f'C{i}' for i in range(len(result_df))]
        result_df.index = cutting_pattern_names
        self.df = result_df


    @property
    def part_count_tall(self) -> pd.DataFrame:
        C_cp = self.part_count_df
        first_non_zero_part = C_cp.idxmax(axis=1)
        # Create a new DataFrame from this Series
        C_cp_tall = first_non_zero_part.reset_index()
        C_cp_tall.columns = ['Cc', 'Part']
        return C_cp_tall

    @property
    def cutting_pattern_names(self)->list[str]:
        return list(self.df.index)


    def generate_matrices(self):

        df = self.df[['element_group_id', 'part_id']]

        part_columns = df['part_id'].unique()
        part_count_df = pd.DataFrame(0, index=df.index, columns=part_columns)
        for index, row in df.iterrows():
            part_count_df.at[index, row['part_id']] += 1

        #C_cp
        self.part_count_df = part_count_df

        # Creating the Second DataFrame (Board Count DataFrame)
        board_columns = df['element_group_id'].unique()
        element_count_df = pd.DataFrame(0, index=df.index, columns=board_columns)
        for index, row in df.iterrows():
            element_count_df.at[index, row['element_group_id']] += 1
        #C_ce
        self.element_count_df = element_count_df    

        #C_cw
        self.waste_df = self.df[['waste']]

        #self.cutting_patterns = count_df


    def generate_matrices_no_usable_residual(self):

        df = self.df[['element_group_id', 'part_id']]
        part_names = self.unique_parts.index
        frequency_matrix = []
        
        for index, row in df.iterrows():
            frequency_row = {part: 0 for part in list(part_names)}
            for part in row['part_id']:
                frequency_row[part] += 1
            frequency_matrix.append({'index': index, 'element_group_id': row['element_group_id'], **frequency_row})

        part_count_df = pd.DataFrame(frequency_matrix)
        part_count_df.index = part_count_df['index']
        part_count_df.index.name = None
        #C_cp
        self.part_count_df = part_count_df[part_names].copy()

        # Creating the Second DataFrame (Board Count DataFrame)
        board_columns = df['element_group_id'].unique()
        element_count_df = pd.DataFrame(0, index=df.index, columns=board_columns)
        for index, row in df.iterrows():
            element_count_df.at[index, row['element_group_id']] += 1
        #C_ce
        self.element_count_df = element_count_df    

        #C_cw
        self.waste_df = self.df[['waste']]

        #self.cutting_patterns = count_df


    def rename_patterns(self):
        new_names = [f'{a}^{b}' for a,b in zip(list(self.df.index),list(self.df['element_group_id']))]
        name_mapper = dict(zip(list(self.part_count_df.index),new_names))
        self.part_count_df.rename(index =name_mapper ,inplace=True)
        self.df.rename(index =name_mapper, inplace=True)   
        self.element_count_df.rename(index =name_mapper ,inplace=True)         
        self.waste_df = self.waste_df.rename(index =name_mapper)         




        ...

    @property
    def possible_lengths(self)->list[float]:
        return list(self.unique_parts.values)

    @property
    def all_element_group_ids(self):
        """TODO"""
        return list(self.inventory["element_group_id"])



@dataclass
class AssemblyPatterns:
    """TODO"""
    
    jointing_patterns: MemberJointingPatterns
    cutting_patterns: ElementCuttingPatterns

    df: Optional[pd.DataFrame] = None
    jointing_patterns_per_assembly: Optional[pd.DataFrame]  = None
    cutting_patterns_per_assembly: Optional[pd.DataFrame]  = None

    def __post_init__(self):
        self.generate_assembly_patterns()
        self.generate_matrices()

    def generate_assembly_patterns(self):
        #parts used per cutting pattern 
        # NOTE: APPLIES TO HEURISTIC 1 PART PER CUTTING PATTERN
        C_cp_tall = self.cutting_patterns.part_count_tall
        #parts used per jointing pattern
        J_jp_by_pos = self.jointing_patterns.part_count_by_pos        
        J_jp_by_pos['Jj'] = J_jp_by_pos.index

        result = J_jp_by_pos.copy()
        component_cols = []
        max_parts = self.jointing_patterns.max_parts
        # Loop through each position index and perform the merge
        for i in range(max_parts):
            pos_col = f'Pos{i}'
            component_cols.append(f'Cc_Pos{i}')
            # Perform the merge based on the position column
            # Make sure C_cp_tall contains a 'Part' column that matches with Pos0, Pos1, etc.
            result = result.merge(C_cp_tall.add_suffix(f'_Pos{i}'), left_on=pos_col, right_on=f'Part_Pos{i}', how='left', suffixes=('', f'_Pos{i}'))

        result=result[['Jj']+component_cols]

        ##############
        ### Replace nans due to '-' (no part required) in certain pos
        ##############
        pos_cols = [col for col in J_jp_by_pos.columns if col.startswith('Pos')]
        # First, identify '-' in df1 position columns and create a boolean mask
        #mask = J_jp_by_pos[pos_cols].eq('-').any(axis=1)
        dash_mask = J_jp_by_pos[pos_cols].eq('-')

        # Merge this mask information into df2
        df1_masked = J_jp_by_pos[['Jj']].join(dash_mask)
        result = result.merge(df1_masked, on='Jj', how='left')

        # Replace NaNs in Cc_Pos columns with '-' where 'has_dash' is True in df2
        for i in range(len(pos_cols)):  # Assuming there are exactly three position columns, adjust range as needed
            pos_col = f'Pos{i}'
            cc_pos_col = f'Cc_Pos{i}'
            # Replace NaN in df2 where corresponding position in df1 has a dash
            result.loc[result[pos_col] & result[cc_pos_col].isna(), cc_pos_col] = '-'

        # Drop the columns used for the mask after processing
        result.drop(pos_cols, axis=1, inplace=True)

        #############
        #remove options which are still n/a (= parts didn't exist in the cutting pattern)
        #############
        result=result.dropna(subset=component_cols, how='any')

        #sort parts (NOTE - ASSUMES ORDER NOT IMPORTANT)
        cc_pos_columns = [col for col in result.columns if col.startswith('Cc_Pos')]
        #result.fillna('!', inplace=True)
        sorted_cols = pd.DataFrame(np.sort(result[cc_pos_columns].values, axis=1), index=result.index)
        sorted_cols.columns = [f'Sorted_Pos{i}' for i in range(sorted_cols.shape[1])]
        sorted_cols.replace('-', np.nan, inplace=True)

        # Drop the original position columns and concatenate sorted columns
        result = pd.concat([result.drop(cc_pos_columns, axis=1), sorted_cols], axis=1)

        # Rename sorted columns back to original pattern if needed
        for i, col in enumerate(sorted_cols.columns):
            result.rename(columns={col: f'Cc_Pos{i}'}, inplace=True)

        # Drop duplicates based on all columns
        result.drop_duplicates(inplace=True)

        #add index
        result.index = [f'A{i}' for i in range(len(result))]
        print(result)
        self.df = result

    def generate_matrices(self):
        
        #get A_aj
        A_aj = pd.get_dummies(self.df['Jj']).rename(columns=lambda x: x.split('_')[-1])
        A_aj = A_aj.astype(int)
        self.jointing_patterns_per_assembly = A_aj

        #get A_ac
        # Collect all column names that match 'Cc_Pos'
        pos_columns = [col for col in self.df.columns if col.startswith('Cc_Pos')]

        # Convert all position columns to dummies and sum them up
        A_ac = pd.DataFrame(index=self.df.index)
        for col in pos_columns:
            A_ac = A_ac.add(pd.get_dummies(self.df[col]), fill_value=0)

        # Ensure all expected components exist even if some components are missing in the dummies
        expected_components = list(A_ac.columns)
        for component in expected_components:
            if component not in A_ac.columns:
                A_ac[component] = 0

        # Reorder the columns according to the expected components
        A_ac = A_ac[expected_components]
        self.cutting_patterns_per_assembly = A_ac.astype(int)


    def rename_patterns(self,member_name: str):
        new_names = [f'{n}^{member_name}' for n in list(self.df.index)]
        name_mapper = dict(zip(list(self.df.index),new_names))
        self.df.rename(index =name_mapper, inplace=True)   
        self.jointing_patterns_per_assembly.rename(index =name_mapper ,inplace=True)
        self.cutting_patterns_per_assembly.rename(index =name_mapper ,inplace=True)         

