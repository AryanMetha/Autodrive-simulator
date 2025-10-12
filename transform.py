import pandas as pd

coordinates = pd.read_csv("/home/aryan/workspaces/new_ws/src/autodrive_test/new_map_2_modified.csv")
coordinates = pd.DataFrame(coordinates)

dx, dy = 5.2492, 0.4043


coordinates['x'] = coordinates['x'] + dx
coordinates['x']=coordinates['x'].round(4)
coordinates['y'] = coordinates['y'] + dy
coordinates['y']=coordinates['y'].round(4)

# Save the result to a new CSV if you wish:
coordinates.to_csv("/home/aryan/workspaces/new_ws/src/autodrive_test/new_map_2_modified_translated.csv", index=False)
