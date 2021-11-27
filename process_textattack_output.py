import pandas as pd
import argparse
import json

TEXT_KEYS = ['premise', 'hypothesis']  # keys used in the text input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--textattack_csv", default=None, type=str, required=True)
    parser.add_argument("--save_filepath", default=None, type=str)
    parser.add_argument('--process_text_split',
                        action='store_true',
                        help='If there is a process step for text, ie split and save as separate columns')
    parser.add_argument('--save_lines', action='store_true')
    args = parser.parse_args()

    # load the textattack output csv as a dataframe
    df = pd.read_csv(args.textattack_file)

    # query the successful attacks
    attacked_df = df.loc[df["result_type"] == "Successful"]

    # separate the original text from the adversarial text
    original_df = attacked_df[["original_text", "original_score", "original_output", "ground_truth_output"]]
    perturbed_df = attacked_df[["perturbed_text", "perturbed_score", "perturbed_output", "ground_truth_output"]]

    # rename columns
    original_df = original_df.rename(columns={"original_text": "text", "original_score": "score", "original_output": "output"})
    perturbed_df = perturbed_df.rename(columns={"perturbed_text": "text", "perturbed_score": "score", "perturbed_output": "output"})
    original_df['attacked'] = 0  # set unmodified text as 0
    perturbed_df['attacked'] = 1  # set perturbed text as 1

    # combine dataframes
    combined_df = pd.concat([original_df, perturbed_df], ignore_index=True)

    # Process text columns
    if args.process_text_split:
        # split according to <split> token
        combined_df[TEXT_KEYS] = combined_df.text.str.split('<SPLIT>', len(TEXT_KEYS) - 1, expand=True)

        for p_text_key in TEXT_KEYS:
            # grab text after the key
            combined_df[p_text_key] = combined_df[p_text_key].str.split(': ', 1, expand=True)[1]
            # remove extra brackets
            combined_df[p_text_key] = combined_df[p_text_key].str.replace('[[', '', regex=False).str.replace(']]', '', regex=False)

    # pull out the dataframe of info to save
    json_df = combined_df[TEXT_KEYS + ["score", "output", "ground_truth_output", "attacked"]]

    # create the JSON string
    result = json_df.to_json(orient="records")

    # save as JSON or JSONL format
    if args.save_lines:
        parsed = json.loads(result)
        with open(args.save_filepath, 'w') as outfile:
            for entry in parsed:
                json.dump(entry, outfile)
                outfile.write('\n')
    else:
        with open(args.save_filepath, 'w') as outfile:
            json.dump(result, outfile)

    print("Conversion complete. File saved at: ", args.save_filepath)
