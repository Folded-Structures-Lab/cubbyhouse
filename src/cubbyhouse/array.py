"""TODO"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from enum import Enum

import numpy as np
from numpy import arange
import pandas as pd

# from math import floor
# from math imort
from cubbyhouse.utils import subset_sum_with_repetition_and_order


class SpanContinuity(Enum):
    """constant class"""
    SINGLE = 1
    DOUBLE = 2


@dataclass_json
@dataclass
class DiscreteJoints:
    """TODO"""

    target_length: int
    max_member_span: int
    min_member_span: int

    # TODO rename some attributes in here as they no longer relate to member span

    spacings: list[float] = field(init=False, repr=False)
    # all_spans_by_support_ID
    # all_spans_by_length

    def __post_init__(self) -> None:
        self.spacings = self.calculate_spacing()

    def calculate_spacing(self) -> list[float]:
        """
        Calculate the spacing of connection points (member spacing) for the total
        length to be spanned.

        Returns:
            list[float]: A list of calculated spacing.
        """

        ordinates = self.ordinates
        spacings = [x-y for x,y in zip(ordinates[1:], ordinates[:-1])]
        
        # extra_length = self.target_length % self.max_member_span
        # num_max_spacing, _ = divmod(
        #     self.target_length - extra_length, self.max_member_span
        # )
        # num_left_spacing = num_right_spacing = 0

        # if extra_length == 0:
        #     # regular member span makes up total length
        #     odd_span = None
        # elif extra_length >= self.min_member_span:
        #     # regular member spacing + one odd length
        #     num_left_spacing = 1
        #     odd_span = extra_length
        # else:
        #     # regular member spacing + two odd lengths
        #     num_left_spacing = 1
        #     num_right_spacing = 1
        #     num_max_spacing -= 1
        #     odd_span = (self.max_member_span + extra_length) / 2

        # return (
        #     num_left_spacing * [odd_span]
        #     + num_max_spacing * [self.max_member_span]
        #     + num_right_spacing * [odd_span]
        # )

        # spacings = list(range(0,self.target_length, self.max_member_span))
        # if spacings[-1]<self.target_length:
        #     spacings.append(self.target_length)

        return spacings
                        

    def as_dataframe(self, row_name: str | None = None) -> pd.DataFrame:
        df = pd.DataFrame([self.ordinates])
        df.columns = [f"joint_{col}" for col in df.columns]
        df = df.rename(index={0: row_name})
        return df

    @property
    def number_of_spans(self) -> int:
        """TODO"""
        return len(self.spacings)

    @property
    def ordinates(self) -> list[float]:
        # """TODO"""
        # ...
        # return [0] + list(np.cumsum(self.spacings))

        ordinates = arange(0,self.target_length, self.max_member_span).tolist()
        if ordinates[-1]<self.target_length:
            ordinates.append(self.target_length)
        return ordinates


    @property
    def ids(self) -> list[int]:
        """TODO"""
        return list(range(len(self.ordinates)))
    
    # @property
    def min_part_length(self, span_type) -> float:
        if span_type == 'single' or len(self.spacings) == 1:
            #single span OR continuous span requiring full length board (end joints only)
            return min(self.spacings)
        if span_type == "continuous":
            pair_sums = [self.spacings[i] + self.spacings[i+1] for i in range(len(self.spacings)-1)]
            
            return min(pair_sums)
                

    @classmethod
    def from_length(cls, target_length: int) -> DiscreteJoints:
        return cls(target_length, target_length,target_length)

    @classmethod
    def from_support_span(cls, target_length: int, support_span: int) -> DiscreteJoints:
        #TODO - include minimum member span?
        return cls(target_length, support_span,support_span)


    # @classmethod
    # def from_(cls, target_length: int, support_span: int) -> DiscreteJoints:
    #     #TODO - include minimum member span?
    #     return cls(target_length, support_span,support_span)




@dataclass
#MEMBER JOINTING PATTERN
class MemberJointingPatterns:
    """TODO"""

    support: DiscreteJoints
    continuity: SpanContinuity

    part_positions: list[tuple] = field(init=False, repr=False)
    # part_lengths
    #unique_lengths: list[float] | None = None
    unique_parts: pd.Series | None = None

    # position_combos
    # part_combos

    max_parts: float = np.inf
    enforce_length_order: bool = False

    unique_length_combo_df: pd.DataFrame = field(init=False, repr=False)
    #parts_to_assemblies_df: pd.DataFrame = field(init=False, repr=False)
    part_count_df: pd.DataFrame = field(init=False, repr=False)
    #members_per_joint_pattern added by member method
    members_per_joint_pattern: pd.DataFrame | None = None   
    part_multiplier: int = 1


    def __post_init__(self) -> None:
        self.part_positions = self.find_part_positions()
        # self.length_part_positions = self.calculate_length_part_positions()
        self.part_lengths, _, self.unique_parts = self.calculate_part_lengths()

        self.position_combos = self.calculate_position_combos_for_total_length()
        self.part_combos = self.calculate_part_combos()
        #same data stored in two different formats:
        # unique_length_combo_df - ordered part lengths
        # part_count_df = part frequency matrix
        self.unique_length_combo_df, self.part_count_df = self.calculate_unique_part_combos()


        self.part_count_by_pos
        # ...
        
        # self.start_pos = self.get_start_pos()
        # self.end_pos = self.get_end_pos()

    def find_part_positions(self):  # -> Tuple[list[Tuple[int, int]], int]:
        """TODO"""
        # support_IDs = self.support.ids
        # n = len(support_IDs)

        if self.continuity in [SpanContinuity.DOUBLE, 'continuous']:
            part_positions = [
                pair
                for pair in itertools.combinations(self.support.ids, 2)
                if abs(pair[0] - pair[1]) > 1
            ]
            # number_of_part_positions = (n * (n - 3)) // 2
        elif self.continuity in [SpanContinuity.SINGLE, 'simple', 'single']:
            part_positions = list(itertools.combinations(self.support.ids, 2))
            # number_of_part_positions = n * (n - 1) // 2
        else:
            raise ValueError(f'span continuity {self.continuity} undefined')
            # number_of_part_positions = n * (n - 1) // 2

        # #note on number of part_positions:
        # #there are n-1 adjactent part_positions
        # #number_of_part_positions = (n * (n - 1) / 2) - (n - 1) -> simplified to (n * (n - 3)) / 2

        return part_positions  # , number_of_part_positions

    def update_for_transverse_laminations(self, transverse_laminations: int) -> None:
        """multiples the number of parts per jointing pattern by transverse_laminations"""
        self.part_multiplier = transverse_laminations
        self.part_count_df = self.part_count_df * transverse_laminations

    def calculate_part_lengths(self):  # -> list[Tuple[float, float]]:
        """TODO"""
        span_lengths = self.support.ordinates
        length_part_positions = [
            (span_lengths[i], span_lengths[j]) for i, j in self.part_positions
        ]
        part_lengths = [j - i for i, j in length_part_positions]
        unique_lengths = sorted(set(part_lengths))

        part_names = [f'P{i}' for i in range(len(unique_lengths))]
        unique_parts = pd.Series(data=unique_lengths, index=part_names)


        return part_lengths, unique_lengths, unique_parts

    def calculate_position_combos_for_total_length(self):
        """TODO"""
        ### FIND ALL POSSIBLE WAYS TO MAKE UP THE TOTLA LENGTH GIVEN THE DISCRETE SPAN LENGTHS
        # #(e.g. pos 2 + 2 + 2 or 4 + 2 = 6)
        s = self.support.number_of_spans
        if self.continuity in [SpanContinuity.DOUBLE, 'continuous']:
            arr = list(range(2, s + 1))
        elif self.continuity in [SpanContinuity.SINGLE, 'simple', 'single']:
            arr = list(range(1, s + 1))
        else:
            raise ValueError(f'continuity {self.continuity} not found')
        target = s

        result = subset_sum_with_repetition_and_order(arr, target)

        # convert to dataframe with position pairs
        index_pos = [[0] + np.cumsum(a).tolist() for a in result]
        position_combos = [
            [
                (index_pos[i][j], index_pos[i][j + 1])
                for j in range(len(index_pos[i]) - 1)
            ]
            for i in range(len(index_pos))
        ]

        return position_combos

    def calculate_part_combos(self) -> list[list[float]]:
        """calculate part (length) combinations based on position combinations"""
        d = self.pos_id_mapper
        part_combos = []
        for combo in self.position_combos:
            part_list = []
            for part in combo:
                part_list.append(d[str(part)])
            part_combos.append(part_list)
        return part_combos

    def calculate_unique_part_combos(self) -> pd.DataFrame:
        """get unique part combinations, considering whether or not the order of parts
        is important)"""
        length_list = self.part_combos
        length = max(map(len, length_list))
        length_array = np.array([x + [np.nan] * (length - len(x)) for x in length_list])
        # sorted_lengths = np.sort(length_array)
        # unique_lengths = np.unique(sorted_lengths, axis=0)

        if self.enforce_length_order:
            raise NotImplementedError("have not enforced the length order")
            # unique_length_combo_df = pd.DataFrame(length_array)
        else:
            # length order doesn't matter - sort by length and drop duplicates
            sorted_lengths = np.sort(length_array)
            unique_length_combo_df = pd.DataFrame(sorted_lengths).drop_duplicates()
            #note - length order doesn't matter for parts being chopped e.g. 300,400 -> same parts
            #length order does matter for reassembling parts at the end
            #index original length_combo_df to preserve order
            unique_index = unique_length_combo_df.index 
            unique_length_combo_df = pd.DataFrame(length_array[unique_index])
            unique_length_combo_df.index=unique_index
            # ...


        df = unique_length_combo_df.copy()

        unique_length_combo_df["num_boards"] = unique_length_combo_df.count(
            axis=1, numeric_only=True
        )
        unique_length_combo_df.sort_values(by="num_boards", inplace=True)

        unique_length_combo_df = unique_length_combo_df[
            unique_length_combo_df["num_boards"] <= self.max_parts
        ]
        # remove extra columns with all nan -> when max_parts > max_parts
        unique_length_combo_df.dropna(axis=1, how="all", inplace=True)

        # unique_length_combo_df
        # rename columns
        unique_length_combo_df.columns = [
            f"part_length_{col}" if col != "num_boards" else col
            for col in unique_length_combo_df.columns
        ]
        #sort to match index order of count_df 
        unique_length_combo_df.sort_index(inplace=True)

        #make matrix format of part to assembly assignment    
        
        # Create the count DataFrame with zeros
        count_df = pd.DataFrame(columns=self.unique_parts.index)

        # Populate the count DataFrame
        for part_id, part_length in self.unique_parts.items():
            count_df[part_id] = df.apply(lambda row: (row == part_length).sum(), axis=1)


        count_df = count_df[count_df.sum(axis=1)<=self.max_parts]
        jointing_names = [f'J{i}' for i in range(len(count_df))]
        count_df.index = jointing_names
        unique_length_combo_df.index=jointing_names

        jointing_patterns = count_df
        

        return unique_length_combo_df, jointing_patterns

    def rename_parts(self, new_parts: pd.Series, rename_dict:str):
        self.unique_parts = new_parts
        self.part_count_df.rename(columns=rename_dict, inplace=True)
        ...

    def rename_patterns(self, member_name: str) -> None:
        new_names = [f'{n}^{member_name}' for n in list(self.part_count_df.index)]
        name_mapper = dict(zip(list(self.part_count_df.index),new_names))
        self.part_count_df.rename(index =name_mapper ,inplace=True)
        self.unique_length_combo_df.rename(index =name_mapper ,inplace=True)

    @property
    def unique_part_names(self) -> list[str]:
        return list(self.part_count_df.columns)
        
    @property
    def unique_lengths(self) -> list[float]:
        return list(self.unique_parts.values)

    @property
    def df(self) -> pd.DataFrame:
        #property to access J_jp from df (consistent with other pattern classes)
        return self.part_count_df

    @property
    def unique_joint_pattern_names(self) -> list[str]:
        return list(self.part_count_df.index)
        

    @property
    def unique_part_names_used_in_jointing_patterns(self) -> list[str]:
        parts_used_in_member_jointing_patterns = self.part_count_df.sum().astype(bool)
        parts_used_in_member_jointing_patterns = list(parts_used_in_member_jointing_patterns[parts_used_in_member_jointing_patterns].index)
        return parts_used_in_member_jointing_patterns
        

    @property
    def part_positions_as_str(self) -> list[str]:
        """TODO"""
        return [str(x) for x in self.part_positions]

    @property
    def pos_id_mapper(self) -> dict:
        """TODO"""
        return dict(zip(self.part_positions_as_str, self.part_lengths))

    @property
    def part_count_by_pos(self) -> pd.DataFrame:
        #OLD METHOD USING part_count_df -> breaks part order
        # data = self.part_count_df
        # max_parts = self.max_parts
        # # Stack the DataFrame to work with multi-level series
        # stacked = data.stack()
        # # Filter out zeros
        # filtered = stacked[stacked > 0]
        # # Reset index to make 'level_1' (the original column names) accessible as a separate column
        # reset = filtered.reset_index(name='Count')
        # # Using apply to repeat rows based on 'Count' and get only the first 'max_parts' entries
        # repeated = reset.loc[reset.index.repeat(reset['Count'])].groupby('level_0').head(max_parts)
        # # Pivot to create the final 'max_parts' position columns
        # pivot = repeated.pivot_table(index='level_0', columns=repeated.groupby('level_0').cumcount(), values='level_1', aggfunc='first')
        # # Fill NaNs with '-' and rename columns
        # final_result_old = pivot.fillna('-').rename(columns=lambda x: f'Pos{x}')

        #NEW METHOD USING unique_length_combo_df -> keeps order
        data = self.unique_length_combo_df.copy()
        data.drop(columns=['num_boards'],inplace=True)
        new_cols = [f'Pos{x}' for x in range(self.max_parts)]
        #NOTE - max_parts could be greater than actual max_parts from unique_length_combo
        #if so, reduce the number of new_cols before assignment
        if len(new_cols)>len(data.columns):
            new_cols = new_cols[:len(data.columns)]
        data.columns=new_cols

        part_length_to_id_mapper = pd.Series(self.unique_parts.index.values, index=self.unique_parts).to_dict()
        data.replace(part_length_to_id_mapper,inplace=True)
        data.fillna('-',inplace=True)
        final_result=data
        return final_result

    @property
    def n_Pm(self):
        return len(self.unique_part_names_used_in_jointing_patterns)

    @property
    def n_Jm(self):
        return len(self.part_count_df)


def main():
    """MAIN"""

if __name__ == "__main__":
    """MAIN"""
    main()
