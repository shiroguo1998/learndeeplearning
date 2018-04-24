# -*- coding: utf-8 -*-
"""
Date: 8/4/2018
Purpose: This file is used to visualize data to get a better understanding of the dataset
Author: Quach Phuong Toan
"""

def check_dead(array):
    if(array[1] == 1):
        return 1
    else:
        return 0


#import libraries
import pandas as pd

#import dataset
gender_submission = pd.read_csv('./Data/gender_submission.csv')
train_data = pd.read_csv('./Data/train.csv')
test_data = pd.read_csv('./Data/test.csv')

#converting Dataframe to numpy array
train_data_np = pd.DataFrame.as_matrix(train_data)

"""
    counting the amount of survival depending ticket
    to check if ticket class has any relationship with survival rate
"""
ticket_survival = {
    "first_class_survive": 0, 
    "second_class_survive": 0, 
    "third_class_survive": 0,
    "first_class": 0,
    "second_class": 0,
    "third_class": 0
}

for i in range(len(train_data_np)):
    if(train_data_np[i, 2] == 1):
        ticket_survival["first_class"] = ticket_survival["first_class"] + 1
        ticket_survival["first_class_survive"] = ticket_survival["first_class_survive"] + check_dead(train_data_np[i])
    elif (train_data_np[i, 2] == 2):
        ticket_survival["second_class"] = ticket_survival["second_class"] + 1
        ticket_survival["second_class_survive"] = ticket_survival["second_class_survive"] + check_dead(train_data_np[i])
    else: 
        ticket_survival["third_class"] = ticket_survival["third_class"] + 1
        ticket_survival["third_class_survive"] = ticket_survival["third_class_survive"] + check_dead(train_data_np[i])
        
print("First class total: " + str(ticket_survival["first_class"]) + "; survive: " + str(ticket_survival["first_class_survive"]))
print("Second class total: " + str(ticket_survival["second_class"]) + "; survive: " + str(ticket_survival["second_class_survive"]))
print("Third class total: " + str(ticket_survival["third_class"]) + "; survive: " + str(ticket_survival["third_class_survive"]))

print("Ratio of first class surviving: ", ticket_survival["first_class_survive"] / ticket_survival["first_class"])
print("Ratio of second class surviving: ", ticket_survival["second_class_survive"] / ticket_survival["second_class"])
print("Ratio of third class surviving: ", ticket_survival["third_class_survive"] / ticket_survival["third_class"])

"""
    counting the amount of survival depending gender
    to check if gender has any relationship with survival rate
"""
gender_survival = {
    "male_survive": 0, 
    "female_survive": 0, 
    "male_total": 0,
    "female_total": 0,
}

for i in range(len(train_data_np)):
    if(train_data_np[i, 4] == "male"):
        gender_survival["male_total"] = gender_survival["male_total"] + 1
        gender_survival["male_survive"] = gender_survival["male_survive"] + check_dead(train_data_np[i])
    elif (train_data_np[i, 4] == "female"):
        gender_survival["female_total"] = gender_survival["female_total"] + 1
        gender_survival["female_survive"] = gender_survival["female_survive"] + check_dead(train_data_np[i])
        
        
print("Male in total: " + str(gender_survival["male_total"]) + "; survive: " + str(gender_survival["male_survive"]))
print("Female in total: " + str(gender_survival["female_total"]) + "; survive: " + str(gender_survival["female_survive"]))

print("Ratio of male surviving: ", gender_survival["male_survive"] / gender_survival["male_total"])
print("Ratio of female surviving: ", gender_survival["female_survive"] / gender_survival["female_total"])

