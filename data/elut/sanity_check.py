import os
import csv

# intact_complex_merge_20230309.test_ppis.txt
complexFile = "../ppi/intact_complex_merge_20230309.test_ppis.txt"

# Loop over .elut files in current directory
for elutFilename in os.listdir():
    if elutFilename.endswith(".elut"):
        curr_elut_prots = set()
        with open(elutFilename, 'r') as elutFile:
            elutReader = csv.reader(elutFile, delimiter='\t')
            next(elutReader)
            for elutRow in elutReader:
                rowSum = sum([float(i) for i in elutRow[1::]])
                if rowSum >= 10.0:
                    curr_elut_prots.add(elutRow[0])

        # Loop over complexes in positive PPIs test file
        with open(complexFile, 'r') as compFile:
            compReader = csv.reader(compFile, delimiter='\t')

            pos_ppi_ct = 0
            for comp in compReader:
                compSet = set(comp)

                if len(compSet.intersection(curr_elut_prots)) == 2:
                    pos_ppi_ct += 1

        print(f"{elutFilename}\n# Positive PPIs: {pos_ppi_ct}\n")

                
