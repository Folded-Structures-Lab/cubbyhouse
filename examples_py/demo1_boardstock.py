
from cubbyhouse.boardstock import BoardStock
from cubbyhouse.utils import get_stock_file_from_case_id

case_stock = get_stock_file_from_case_id("case_2")
board_stock = BoardStock.from_element_group_csv(case_stock)

print("\n Element Groups:")
print(board_stock.element_group_df)
print("\n Section Groups:")
print(board_stock.section_group_df)
