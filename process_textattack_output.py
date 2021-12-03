import pandas as pd
import numpy as np
import argparse
import json

import os
import os.path as osp


TEXT_KEYS = ['Premise', 'Hypothesis']  # keys used in the text input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--textattack_csv", default=None, type=str, required=True)
    parser.add_argument("--save_filepath", default=None, type=str)
    parser.add_argument('--process_text_split',
                        action='store_true',
                        help='If there is a process step for text, ie split and save as separate columns')
    parser.add_argument('--save_lines', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_filepath, exist_ok=True)

    # load the textattack output csv as a dataframe
    df = pd.read_csv(args.textattack_csv)

    # query the successful attacks
    attacked_df = df.loc[df["result_type"] == "Successful"]

    # split data into train, val, test
    train, validate, test = \
              np.split(attacked_df.sample(frac=1, random_state=42), 
                       [int(.8*len(attacked_df)), int(.9*len(attacked_df))])
    data_splits = {
        "train": train,
        "val": validate,
        "test": test
    }

    all_df = []

    for split in data_splits.keys():
        print("Processing split:", split)
        attacked_df = data_splits[split]

        # separate the original text from the adversarial text
        original_df = attacked_df[["original_text", "original_score", "original_output", "ground_truth_output"]]
        perturbed_df = attacked_df[["perturbed_text", "perturbed_score", "perturbed_output", "ground_truth_output"]]

        # rename columns
        original_df = original_df.rename(columns={"original_text": "text", "original_score": "score", "original_output": "output"})
        perturbed_df = perturbed_df.rename(columns={"perturbed_text": "text", "perturbed_score": "score", "perturbed_output": "output"})
        original_df['label'] = 0  # set unmodified text as 0
        perturbed_df['label'] = 1  # set perturbed text as 1

        # combine dataframes
        combined_df = pd.concat([original_df, perturbed_df], ignore_index=True)
        combined_df['text'] = combined_df['text'].str.replace('[[', '', regex=False).str.replace(']]', '', regex=False)

        # Process text columns
        if args.process_text_split:
            # remove square brackets and replace <split> token with [sep]
            combined_df['text'] = combined_df['text'].str.replace('[[', '', regex=False).str.replace(']]', '', regex=False).str.replace('<SPLIT>', ' [SEP] ', regex=False)

            for p_text_key in TEXT_KEYS:
                # grab text without the key
                combined_df['text'] = combined_df['text'].str.replace(p_text_key + ': ', '', regex=False)
        
        # FILTER
        combined_df = combined_df[combined_df['text'].str.split().str.len() <=512]

        # pull out the dataframe of info to save
        json_df = combined_df[["text", "score", "output", "ground_truth_output", "label"]]
        json_df['ex_id'] = json_df.index

        # create the JSON string
        result = json_df.to_json(orient="records")

        # save as JSON or JSONL format
        if args.save_lines:
            out_filepath = osp.join(args.save_filepath, split + ".jsonl")
            parsed = json.loads(result)
            with open(out_filepath, 'w') as outfile:
                for entry in parsed:
                    json.dump(entry, outfile)
                    outfile.write('\n')
        else:
            out_filepath = osp.join(args.save_filepath, split + ".json")
            with open(out_filepath, 'w') as outfile:
                json.dump(result, outfile)
        all_df.append(json_df)

        print("Conversion complete. File saved at: ", out_filepath)
        print("Split", split, "has length", json_df.shape[0])

    # count maximum words
    combined_df = pd.concat(all_df, ignore_index=True)
    count = combined_df['text'].str.split().str.len()
    count.index = count.index.astype(str) + ' words:'
    count.sort_index(inplace=True)
    print(count)
    print("Maximum length in words:", count.max())
    print("Average length in words:", count.mean())
