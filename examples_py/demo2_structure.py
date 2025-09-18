
from cubbyhouse.building import (
    TimberFramedStructure,
    FramingGroup,
    DeckingParams,
    DeckJoistParams
)

########################
# SETUP FRAMING GROUPS
########################

x_extent = 6070
y_extent = 1953
deck_design_span = 450
joist_design_span = 1200 

#create framing groups
deck_group = FramingGroup(
    "F1",
    length=x_extent,
    width=y_extent,
    design_parameters=DeckingParams(
        design_span=deck_design_span, span_type="continuous"
    )
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

print('\nDeck Framing Group:')
print(deck_group)

# default discrete_joints created based on length 
print('\nDeck Framing Group - discrete joints, spacings, and min part length:')
print(deck_group.discrete_joints)
print(deck_group.discrete_joints.spacings)
print(deck_group.min_part_length)

#create structure
structure = TimberFramedStructure.from_list([deck_group, joist_group])
# print('\nStructure:')
# print(structure)
print('\nDesign spans for all framing groups:')
print(structure.framing_group_design_spans_df)

# update framing group joints
deck_group.update_framing_group_joints(joints_from="design_span")
joist_group.update_framing_group_joints(joints_from="design_span")


print('\nDeck Framing Group - updated discrete joints, spacings, and min part length:')
print(structure.framing_groups['F1'].discrete_joints)
#print(deck_group.discrete_joints) #equivalent to above
print(deck_group.discrete_joints.spacings)
print(deck_group.discrete_joints.ordinates)
print(deck_group.min_part_length)


print(structure.framing_group_df)
