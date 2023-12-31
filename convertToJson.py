import csv
import json

def make_json(csvFilePath, jsonFilePath):

    data = {}

    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        for rows in csvReader:
            # Exclude the 'ID' row directly when creating the dictionary:
            data[rows['RowNum']] = {key: value for key, value in rows.items() if key != 'RowNum'}

    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

# Driver Code
csvFilePath = r"bookings.csv" #add any csv file which needs to be converted to json file
jsonFilePath = r"convert.json" # converted json file
make_json(csvFilePath, jsonFilePath)