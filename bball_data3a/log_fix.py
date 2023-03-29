import json5
from sys import argv
from io import TextIOWrapper
#import json
# import orjson
# import numpy

def clean_ugly_json_file(file_in: str, file_out: TextIOWrapper) -> None:
    """
        Open shitty horrible trash ill-formatted log files and make them something less terrible.
    """
    
    json_in = []
    new_data = []

    with open(file_in,'r') as f:
        for line in f:
            try:
                json_in.append(json5.loads(line))
            except:
                pass

    idx = 0
    out_json = ""

    for i in json_in:
        
        lvl, arr = {}, []
        lvl['idx'] = int(idx)
        lvl['ts'] = int(i['ts'])

        for pt in i['dataFrame']['detected_points'].values():
            interim_arr = [pt['x'], pt['y'], pt['z'], pt['v']]
            arr.append(interim_arr)
            # np.array([pt['x'], pt['y'], pt['z'], pt['v']])
            #arr = np.append(arr, interim_arr, )

        lvl['xyzv'] = arr
        #new_data.append(lvl)
        out_json = out_json + json5.dumps(lvl, quote_keys=True) + ",\n"
        idx += 1
    
    #out_json = json5.dumps(new_data, indent=4, trailing_commas=True)[1:][:-1]
    file_out.write(out_json)


if __name__ == '__main__':
    argv = [i for i in argv if '.py' not in i]

    if len(argv) < 2:
        print("not enough parameters passed. gtfo")
        exit()

    file_out = argv[-1]
    f_out = open(file_out, 'w', buffering=1)
    f_out.write('[\n')

    for file_in in argv[:-1]:
        clean_ugly_json_file(file_in, f_out)
        print(f"Processed {file_in} successfully!")
    
    f_out.write(']')
    print(f"All log files output sent to {file_out} !!")


exit()
