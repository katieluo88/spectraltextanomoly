#!/usr/bin/python

import sys, getopt
import json
import csv


def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    data_str = inputfile
    save_str = outputfile
    
    with open(data_str, 'r') as json_file:
        json_list = list(json_file)

    # now we will open a file for writing
    data_file = open(save_str, 'w')

    # create the csv writer object
    csv_writer = csv.writer(data_file)

    row = 0

    # list of data
    for json_str in json_list:
        results = json.loads(json_str)
        if row == 0:
            header = results.keys()
            print("Writing headers:", header)
            csv_writer.writerow(header)

        # write csv data
        csv_writer.writerow(results.values())
        row += 1

    data_file.close()
    print("Number of rows:", row)

if __name__ == "__main__":
    main(sys.argv[1:])

