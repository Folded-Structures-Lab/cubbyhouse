"""TODO"""

from __future__ import annotations

from dataclasses import dataclass, field
# import json
import pandas as pd
import re

@dataclass
class Board:
    name: str
    size: str | None
    mat: str
    depth: float
    width: float
    length: float

    def __post_init__(self):
        if self.size is None:
            self.size = f"{self.depth}x{self.width}"


@dataclass
class Element:
    name: str
    board_data: dict[str, Board]
    qty: int | None = None

    # @property
    # def attr_id(self) -> str:
    #     return f"{self.size}_{self.mat}_{str(self.length)}"


@dataclass
class ElementGroup:
    name: str
    element_group_data: dict[str, Element]
    qty: int | None = None

    # @property
    # def attr_id(self) -> str:
    #     return f"{self.size}_{self.mat}_{str(self.length)}"


@dataclass
class BoardStock:
    ...

    element_group_df: pd.DataFrame
    # board_df: pd.DataFrame | None = field(repr=False, default=None)
    section_group_df: pd.DataFrame | None = field(repr=False, default=None)

    def __post_init__(self):
        self.add_element_group_names_and_lengths()
        self.create_section_groups()

    def create_section_groups(self) -> None:
        """create section groups from the element_group_df"""
        #create section_group from all element_groups
        self.section_group_df = self.element_groups_to_section_groups(self.element_group_df)

        # Create a mapping from element_group to section_group indices
        section_group_mapping = {
            index: group_index
            for group_index, indices in zip(
                self.section_group_df.index, self.section_group_df["element_group_indices"]
            )
            for index in indices
        }

        # Add the section group indices to the original DataFrame
        self.element_group_df["section_group"] = self.element_group_df.index.map(
            section_group_mapping
        )


    @staticmethod
    def element_groups_to_section_groups(element_group_df: pd.DataFrame) -> pd.DataFrame:
        # Group by 'size' and 'grade', summing 'L_total'
        section_group_df = element_group_df.groupby(
            ["size", "grade"], as_index=False
        ).agg({"L_total": "sum"})

        # Add the element_group indices for each group
        element_group_indices = (
            element_group_df.groupby(["size", "grade"])
            .apply(lambda x: list(x.index))
            .reset_index(name="element_group_indices")
        )

        # Merge the grouped data with the element_group indices
        section_group_df = pd.merge(
            section_group_df, element_group_indices, on=["size", "grade"]
        )

        #sort to be in element order 
        #find first element_group 
        first_element_group = [x[0] for x in element_group_indices['element_group_indices'].to_list()]
        # replace 'E1', 'E10' with 1, 10 etc
        first_element_group = [int(re.sub("[^0-9]", "", x)) for x in first_element_group]
        section_group_df["first_element_group"] = first_element_group
        section_group_df.sort_values(by="first_element_group", inplace=True, ignore_index=True)
        section_group_df.drop("first_element_group",axis=1, inplace=True)


        # Changing index to G1...GN
        section_group_df["index"] = [
            "G" + str(i + 1) for i in range(len(section_group_df))
        ]
        section_group_df.set_index("index", inplace=True)

        section_group_df = section_group_df
        return section_group_df


    @classmethod
    def from_element_group_csv(cls, lib_path: str) -> BoardStock:
        """create a BoardStock from a csv file of element_group"""
        df = pd.read_csv(lib_path)
        board_stock = cls(element_group_df=df)
        return board_stock

    @property
    def unique(self) -> pd.DataFrame:
        """get unique board types (same size and grade) from the inventory"""
        return self.data[["size", "grade"]].drop_duplicates().reset_index(drop=True)

    @property
    def lengths(self) -> pd.DataFrame:
        """get the total length of unique board types"""
        return self.total_lengths_greater_than(0)

    @property
    def element_group_quantities(self) -> pd.Series:
        return self.element_group_df[['qty']]

    @property
    def element_group_names(self) -> list[str]:
        return list(self.element_group_df.index)

    @property
    def section_group_names(self) -> list[str]:
        return list(self.section_group_df.index)


    def add_element_group_names_and_lengths(self) -> None:
        # check element_group_df has name columns
        if "name" not in self.element_group_df.columns:  # add unique name for each element_group
            element_group_names = ["E" + str(index + 1) for index in self.element_group_df.index]
            self.element_group_df.loc[:, "name"] = element_group_names


            # Reorder columns to have "name" first
            cols = ["name"] + [col for col in self.element_group_df.columns if col != "name"]
            self.element_group_df = self.element_group_df.loc[:, cols]

        # Add total lengths
        self.element_group_df.loc[:, "L_total"] = (
            self.element_group_df["L"] * self.element_group_df["qty"]
        )


        self.element_group_df.set_index("name", inplace=True)

    def to_json(self, file_name: str) -> None:
        ...
        self.data.to_json(file_name, orient="records")

    def inv_longer_than(self, min_length) -> pd.DataFrame:
        """get the inventory with length greater than min_length"""
        return self.data[self.data["L"] > min_length]


    def element_group_lengths_greater_than(self, part_length: float) -> pd.DataFrame:
        """get the element_groups with length greater than part_length"""
        return self.element_group_df[self.element_group_df["L"]>=part_length]

    def section_group_lengths_greater_than(self, part_length: float) -> pd.DataFrame:
        """get the total length per section group type,  with length greater than part_length"""
        #set up data frame with all section_groups
        section_group_df = pd.DataFrame(index = self.section_group_df.index, 
                                        columns=self.section_group_df.columns)
        section_group_df["L_total"] = 0
        
        #get element_groups and section_group lengths longer than part_length
        element_groups = self.element_group_lengths_greater_than(part_length)
        if len(element_groups)>0:
            section_group_ok_lengths_df = self.element_groups_to_section_groups(element_groups)
            #update section_group_df
            section_group_df.update(section_group_ok_lengths_df)
        return section_group_df

    def total_lengths_greater_than(self, min_length) -> pd.DataFrame:
        """get the total length per board type,  with length greater than min_length"""
        df_with_ok_lengths = self.inv_longer_than(min_length)

        df_with_ok_lengths["L_times_qty"] = (
            df_with_ok_lengths["L"] * df_with_ok_lengths["qty"]
        )

        # Group by "size" and "grade" and sum the "L_times_qty" column
        result = (
            df_with_ok_lengths.groupby(["size", "grade"])["L_times_qty"]
            .sum()
            .reset_index()
        )

        # Rename the column to "L"
        result.rename(columns={"L_times_qty": "L_total"}, inplace=True)

        return result
