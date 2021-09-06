import pyfiglet
import os
import json
from pprint import pprint


file_input_list = [
    'D:/Datasets/Kaggledatasets/TweetsChampions.json'
]

combined_file_name = 'TweetsChampions.json'
combined_top_key = 'tweet'

def add_file_of_multiple_dicts_to_one_list(file_in):

    return_obj = []
    with open(file_in) as fp:
        for line in fp:
            if len(line)>5:
                res = json.loads(line)
                return_obj.append(res)
    return return_obj



list_res = []
for rs in file_input_list:
    list_res = add_file_of_multiple_dicts_to_one_list(rs)
#pprint(list_res)\
with open(combined_file_name, 'w') as outfile:
    json.dump(list_res, outfile)
result = pyfiglet.figlet_format("Do you want to play a game?", font="slant")
print(result)
