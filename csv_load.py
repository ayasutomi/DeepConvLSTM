import numpy as np 
import csv

def csv_load(filename):
    with open(filename) as file:
        data = np.array(list(csv.reader(file,delimiter=',')))
        data = data[1:,1:]
        # print (data)    
    
    return data

csv_load('./data.csv')

