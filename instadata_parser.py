import json
import pickle
from tqdm import tqdm


def parse(annotations_file):
    print('Parsing the ', annotations_file)
    insta = json.load(open(path+annotations_file, 'r'))
    parsed_insta={}
    for up_key, up_val in tqdm(insta.items()):
        for sub_key, sub_val in up_val.items():
            parsed_insta.update({up_key+'_@_'+sub_key: sub_val})
    annotations_file = annotations_file[:-4]
    print('Num examples: ', len(parsed_insta))
    return parsed_insta, annotations_file


if __name__ == "__main__":
    path = './captions/annotations/'
    annotations_files = ['insta-caption-train.json', 
                         'insta-caption-test1.json', 
                         'insta-caption-test2.json']

    parsed_insta, annotations_file = parse(annotations_files[0])                     
    pickle.dump(parsed_insta, open(path+annotations_file+'pkl', 'wb'))

    parsed_insta, annotations_file = parse(annotations_files[1])
    parsed_insta2, _ = parse(annotations_files[2])

    parsed_insta.update(parsed_insta2)
    pickle.dump(parsed_insta, open(path+annotations_file+'pkl', 'wb'))
