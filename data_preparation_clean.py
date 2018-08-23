#Julia Ridley
#Applied Machine Learning
#Final Project data preparation


import os
filepath = "/Users/juliaridley/Google Drive/METALS/Spring 2018/Applied machine learning/final project/data/clean"
os.chdir(filepath)

import pandas as pd

data_cleaned = pd.read_csv("original_data.csv")
orig_data = pd.read_csv("original_data.csv")

###############################################################################
#data_cleaned cleaning and preparation

empty_vars = ["Student Response Subtype", "Tutor Response Subtype", 
              "Feedback Classification", "KC Category (Default)", 
              "KC Category (tutored-only)", 
              "KC Category (tutored-only-description)", 
              "KC Category (skill-tutorONLY)", 
              "KC Category (produce_explain-tutorONLY)", 
              "KC Category (answer-tutorONLY)", "KC Category (Single-KC)", 
              "KC Category (Unique-step)", "Class", "Unnamed: 51", 
              "Unnamed: 52"]

data_cleaned = data_cleaned.drop(empty_vars, axis = 1)

one_val_vars = ["Sample Name", "Time Zone", "Level (Section)", "KC (Single-KC)", 
                "School"]

data_cleaned = data_cleaned.drop(one_val_vars, axis = 1)

uninformative_vars = ["Row", "Step Name", "KC (produce_explain-tutorONLY)", 
                      "Problem Start Time", "Time", "CF (tool_event_time)", "Action.1", 
                      "Input.1", "KC (tutored-only)", "Total Num Hints", 
                      "KC (Unique-step)", "Problem Start Time"]

data_cleaned = data_cleaned.drop(uninformative_vars, axis = 1)

missing_data_vars = ["KC (tutored-only-description)", "KC (skill-tutorONLY)", 
                     "KC (answer-tutorONLY)"]

data_cleaned = data_cleaned.drop(missing_data_vars, axis = 1)

#keeping "Student Response Type" to drop any hint requests first
too_informative_vars = ["Tutor Response Type", "Is Last Attempt", "Selection.1", 
                        "Action", "Feedback Text"]

data_cleaned = data_cleaned.drop(too_informative_vars, axis = 1)

#keeping transaction ID and Anon Student Id as a references, but not using them as features

#rows that are wierd and broken
drop__broken_session_ids = ["ed41cdfd0f0d740fd63c2ba87e0c41ce", 
                            "4a0b5c7ef15d86052b22d8eb38a96cf7", 
                            "b7035297f624f17caa3f3dcb7a172adf", 
                            "5cb032523be20911da1cc74527200c27"]

#rows that have only a period instead of a duration
drop_missing_duration_ids = ["564c96c71df38141e9903e6474453184", 
                             "9ea1c4395a0947e26d388f1381cc33b6", 
                             "251c789cb9d6be10d14255d326587b6f",
                             "71bc82f07a6b9665e1af2b42ef91515d",
                             "dcef09e54a2b0b7f5bfb018ddf1d4d0e"]

#dropping the broken session IDs
for id in drop__broken_session_ids:
    data_cleaned = data_cleaned[data_cleaned["Transaction Id"] != id]

#dropping the missing duration IDs
for id in drop_missing_duration_ids:
    data_cleaned = data_cleaned[data_cleaned["Transaction Id"] != id]

#replacing the Time feature with CF (tool_event_time), which is the time that 
#the response is submitted, rather than the time the section is started

data_cleaned = data_cleaned.rename(index=str, columns={"Duration (sec)": "Step Duration",
                                       "CF (tutor_event_time)": "Time"})

#excluding instances that don't have Outcome
data_cleaned = data_cleaned.dropna(subset=["Outcome"])
#removing survey columns as instances to predict
survey_results = data_cleaned[data_cleaned["Problem Name"] == "survey"]    
data_cleaned = data_cleaned[data_cleaned["Problem Name"] != "survey"]
#dropping any hint requests
data_cleaned = data_cleaned[data_cleaned["Student Response Type"] != "HINT_REQUEST"]
#dropping "Student Response Type" and "Help Level"
data_cleaned = data_cleaned.drop(["Student Response Type", "Help Level"], axis = 1)


#removing rows that are pressing the "done" button
data_cleaned = data_cleaned[data_cleaned["Selection"] != "dorminButton1"]
data_cleaned = data_cleaned[data_cleaned["Selection"] != "done"]

#recoding "articleTutorA" as "articleTutor-A"
data_cleaned["Problem Name"] = data_cleaned["Problem Name"].replace("articleTutorA", "articleTutor-A")


#reordering so that "Outcome" is the last variable
cols = data_cleaned.columns.tolist()
cols.insert(len(cols), cols.pop(cols.index("Outcome")))
data_cleaned = data_cleaned.reindex(columns= cols)

filepath = "/Users/juliaridley/Google Drive/METALS/Spring 2018/Applied machine learning/final project/data/clean/datasets"
os.chdir(filepath)
            
data_cleaned.to_csv("data_cleaned.csv")

###############################################################################
#feature engineering

data_engineered = data_cleaned.copy(deep=True)


#splitting the timestamp into a date and time
new_date = []
new_time = []
for row in data_engineered["Time"]:
    split_time = row.split(" ")
    new_date.append(split_time[0])
    split_time = split_time[1]
    split_time = split_time.rsplit(":", 1)
    split_time = split_time[0]
    new_time.append(split_time)

#creating the new columns from the date and time lists
data_engineered["New Date"] = new_date
data_engineered["New Time"] = new_time
data_engineered = data_engineered.drop("Time", axis=1)

#changing the New Time feature to datetime
data_engineered["New Date"] = pd.to_datetime(data_engineered["New Date"])
data_engineered["New Time"] = pd.to_datetime(data_engineered["New Time"])

#make sure that the data is ordered chronologically
data_engineered = data_engineered.sort_values(by=["Anon Student Id", "New Date", "New Time"])

#create a new variable for sex based on the survey data
id_sex = []        
for index, row in survey_results.iterrows():
    if row["Selection"] == "dorminComboBox1":
        id_sex.append((row["Anon Student Id"], row["Input"]))

#create a new column for sex
id_sex = dict(id_sex)
id_sex = pd.DataFrame(list(id_sex.items()), columns = ["Anon Student Id", "Sex"])
data_engineered = pd.merge(data_engineered, id_sex, on='Anon Student Id', how='left')

#Pankaj code
#creating a new variable for age based on the survey data
id_age = survey_results[survey_results["Selection"] == "dorminTextField1"]

def digit_checker(row):
    for age in row:
        if age.isdigit() == True:
            return 1
        else:
            return 0

id_age["age_num_flag"] = id_age['Input'].apply(digit_checker)

id_age = id_age[id_age["age_num_flag"] == 1]
cols = ['Anon Student Id', 'Input']
id_age = id_age[cols]

id_age.columns = ['Anon Student Id', 'Age']

data_engineered = pd.merge(data_engineered, id_age, on='Anon Student Id', how='left')
data_engineered["Age"] = data_engineered["Age"].astype(float)

#recoding the outcome variable
data_engineered["Outcome"] = data_engineered["Outcome"].replace("CORRECT", 1)
data_engineered["Outcome"] = data_engineered["Outcome"].replace("INCORRECT", 0)


#summing the duration at each step to get the problem duration
data_engineered["Step Duration"] = data_engineered["Step Duration"].astype(float)
data_engineered["Problem Duration"] = data_engineered.groupby(
        ['Anon Student Id', 'Session Id'])["Step Duration"].cumsum()

#summing the duration of the student the entire time
data_engineered["Total Duration"] = data_engineered.groupby(
        ['Anon Student Id'])["Step Duration"].cumsum()

#creating a running percent correct total for each problem
data_engineered["Problem Sum"] = data_engineered.groupby(
        ['Anon Student Id', 'Session Id'])["Outcome"].cumsum()
data_engineered["Problem Count"] = data_engineered.groupby(
        ['Anon Student Id', 'Session Id']).cumcount()+1
data_engineered["Problem PC"] = data_engineered["Problem Sum"]/data_engineered["Problem Count"]
data_engineered = data_engineered.drop(["Problem Sum", "Problem Count"], axis=1)

#creating a running percent correct total for each the entire time
data_engineered["Total Sum"] = data_engineered.groupby(
        ['Anon Student Id'])["Outcome"].cumsum()
data_engineered["Total Count"] = data_engineered.groupby(
        ['Anon Student Id']).cumcount()+1
data_engineered["Total PC"] = data_engineered["Total Sum"]/data_engineered["Total Count"]
data_engineered = data_engineered.drop(["Total Sum", "Total Count"], axis=1)
    

#Pankaj code
#creating a new variable that determines whether the student is answering questions out of order
session_list = list(data_engineered["Session Id"].unique())
counter = 0
def skipper(row):
    if row == 0 :
        return "Same"
    if row == 1 :
        return "Next"
    elif row >1:
        return "Skip Ahead"
    elif row<0:
        return "Skip Back"
    else:
        return "First"
    
for session in session_list:
    
    temp = data_engineered[data_engineered["Session Id"]==session]
    temp.reset_index(inplace=True)
    temp.drop("index",1,inplace=True)
    temp2 = pd.DataFrame(temp.Selection.str.split("Box",1).tolist(),columns = ["start","Boxnum"])
    temp2.drop("start",1,inplace=True)
    temp2 = pd.DataFrame(temp2['Boxnum'].astype(str).astype(int))
    temp = pd.concat([temp, temp2], axis=1)
    
    temp3 = pd.DataFrame(temp.Boxnum.shift(1))
    temp3.columns = ["Boxnum_prev"]
    temp = pd.concat([temp, temp3], axis=1)
    temp["PathDif"] = temp["Boxnum"] - temp["Boxnum_prev"]
    
    temp["Step Order"] = temp['PathDif'].apply(skipper)
    if counter ==0:
        counter +=1
        result = temp
    else:
        result = result.append(temp)

result = result.drop(["Boxnum", "Boxnum_prev", "PathDif"], axis = 1)
data_engineered = result
data_engineered = data_engineered.sort_values(by=["Anon Student Id", "New Date", "New Time"])


#recreating the original outcome variable
data_engineered["Outcome"] = data_engineered["Outcome"].replace(1, "CORRECT")
data_engineered["Outcome"] = data_engineered["Outcome"].replace(0, "INCORRECT")

            
#reordering so that "Outcome" is the last variable
cols = data_engineered.columns.tolist()
cols.insert(len(cols), cols.pop(cols.index("Outcome")))
data_engineered = data_engineered.reindex(columns= cols)   

data_engineered.to_csv("data_engineered.csv",index=False)


###############################################################################
#splitting into dev/dv/test sets

dev_student_ids = ["Stu_75a59a029f91aefb104ef09e04496d03",
                   "Stu_dfcc8d25b3c1c5c1b38599248c59c9ce",
                    "Stu_af7762c2bd6284043d231328da13a3af",
                    "Stu_5c1861e54686f831bbbefbc0db229779",
                    "Stu_80d24c182065611628c517d6ba622d5f",
                    "Stu_628ec16da25dbe8ef1b168a9a64a781e",
                    "Stu_88d5c63c928bc54e50645bed94bc289d",
                    "Stu_cbf326e4cb421ed0c7b14ba5f05d82b5",
                    "Stu_d1ae5d1a17223959dae20f9a62087fd2",
                    "Stu_8ed4fc5d24429ff663494d26cad0dae7",
                    "Test_6349854921f7c468a6d124c1dfe8f29c",
                    "Stu_2a375349f13134798d0856b0262874b8",
                    "Stu_e641e70c1c5940c6340636a03e1ac271",
                    "Stu_266fad41fe7953cf2934f6126101dc43",
                    "Stu_852c2c0072b3f895a893cac8a7636559",
                    "Stu_df0598130825a184c84b5384036c4577",
                    "Stu_b55f8db26552398f393ab5b538df9798",
                    "Stu_0fdf5887d7b2aae29a585a8031b52486",
                    "Test_a93a6cf16347462d5a044ba405afc11c",
                    "Stu_210a9d49ec920124d7a55510eb691ac7",
                    "Stu_c48054c674d42e9b493458df6f08c3b9",
                    "Stu_fca82e64be7b24e420a309b15f0ee483",
                    "Stu_549d0da39520a0330b98b57e12d7309b",
                    "Stu_3310985d8fb73b03b4c9dff6873fdefa",
                    "Stu_d35b3dec9dc5d5b0862e1a3ec16e9abd",
                    "Stu_3f7acb7cf1528d402ea8d977501fe5ff",
                    "Stu_6435babbfcf3511ccfe1cab7a042df0d",
                    "Stu_6d80444fe22f21a64dc1d0dc8af32fef",
                    "Stu_cbbf14b0971adf5418eaa4e9b61a6c51",
                    "Stu_c7d5927ae4ea790cfbebbf18cf579045"]


cv_student_ids = ["Stu_86f467f6e5c6403e9060e2d344d9eb66",
                "Stu_02557b9f86b335cc45cf2a266c5bb900",
                "Stu_5ea80265598415aa8a4c049932fd600a",
                "Stu_f6a1098ec765c451704f6474be2a1e8a",
                "Stu_d73684b50b02741c2b1587e099c1097f",
                "Test_062587e9772411024bac92bd036fde85",
                "Test_d517cea4cbc6cebc7c5b4f403f022241",
                "Test_94aeae0108f54f9dd659fca72a1f374d",
                "Stu_93c1509c4c834285874a5c09adb0eeee",
                "Stu_0d56426b45e48264501e1e83180d5fa9",
                "Stu_72e24051eb26a9eed5599769cc98b8f5",
                "Stu_ce092e79305bf0431ba5014a386d2670",
                "Stu_861cee66db2f516aa4f731b64a40b493",
                "Stu_e8895bd04fdcf89634176e3a039892bb",
                "Test_c126fe3ade562f349bfda66cdef51cff",
                "Stu_478aad31b5004c7fcd828cb12e111312",
                "Test_f9fce3e159b5a7f19ee7dea028f59355",
                "Stu_08d6f18072426ea643fdb186d78bc5fd",
                "Stu_313b827093c11df63594683bdff83ed7",
                "Stu_ab2f97be8e573f157985c60e149da3e8",
                "Test_3cd1400c518d43de8b8f1903d52fc3bb",
                "Stu_21be03fbea37ae282aaea3b5c3f3153c",
                "Stu_ecb24a15d0cb71bd3cd98247aafc3730",
                "Stu_7efab309024653cbf18bb19658bd298c",
                "Stu_af03494fd598a31c25c1b703253836e6",
                "Stu_691524bea9e406041e1e83d19a0d832f",
                "Stu_2a1a17842b899b2207802b169931983f",
                "Stu_5e4327d9db3b7846f1784dc9671595f3",
                "Stu_90007b7a24ad22e5548417c87641a2d6",
                "Stu_1f83eb930e5ee01298c269a5328b6f45",
                "Stu_fe9b28f3ac97588f0a3b3edf9a76b200",
                "Test_c6ed67c702f0311c7a1cc8b13b23542f",
                "Test_9e5e6c4538638280ff95c5eb02192e27",
                "Test_6ee549dea8e6cf16f6238f5c1ada4ee5",
                "Test_c861f09001bcc84fe8ea8225b3a25489",
                "Test_11d0de63acf155f4488f704c82cf594e",
                "Stu_286639b0e6733cd98cde19c119765e95",
                "Stu_cc0598824b037a90d33909cbe7c44d6f",
                "Stu_5127c492845a4dccaff73b160f8745fc",
                "Stu_d0415881d8976560ead24746e30def4f",
                "Stu_78117eb46d304667e5b88dd9c80b4981",
                "Test_6e67303fcd73005cc24cb4b57d04c5a9",
                "Test_90deb6fce34635cf7771f4389f87d14b",
                "Stu_2e476ff183afca8db09bf628b80a58b6",
                "Stu_fb5e8b321745cd837d1c2ad9188f1a2d",
                "Stu_68bb8a12fee185869595c1e246dc748e",
                "Test_1b9ca74f1052ab205beaf4a8e49a4570",
                "Stu_a097a45664679a269651bc93769d2301",
                "Stu_b2038022565752f38f8bc3bfa44fb13f",
                "Stu_5a475823200a0a71d42606faaa39b8a1",
                "Stu_45ca2a6fd64e9d9f73a1214e8045aac9",
                "Stu_7d1e4fb15d64e2fc66176c04872ec907",
                "Stu_790c4a5278585e3b97d6ba86789e9da4",
                "Stu_d8ebff81b31318603049a498b3bcad05",
                "Stu_3aeb54e793c2201de7d453cd1c4ccb27",
                "Stu_2b55705135f4fb6464301e71d0d3c179",
                "Stu_5a671033ca22b6a36d33628f9c63b46d",
                "Stu_091c75fce4cff66def940affed136ec9",
                "Stu_9ac2e0f9139b474c3daeb399c28cedb3",
                "Stu_4096fd7716cb9912881d56e820e9a359",
                "Test_bfd929fa1bd5573c44343f23893f6710",
                "Stu_4ce9aafb07233c8118a8def3c529cca9",
                "Stu_545b3121bd03cf9aa4dd2aaa1dc2f4c9",
                "Stu_fe5fca7d5043dbb03865e7f4657505c0",
                "Test_341041474537dd76783bcc2fbbdd4244",
                "Stu_b4a0a806e7bf82f1c8287c70559c0616",
                "Stu_61c6243ebf079c92ea7780d8211da7a5",
                "Test_8cabae3510d19879174e6417d80b4729",
                "Stu_4096a442a0afa65f8ae67155a78d22a8",
                "Stu_3e2f166947cb4b8a5c84f8eed2ba1627",
                "Test_1d4c64c1afd28ca1f0122561bd997063",
                "Test_e959d01184d140c40935fff0e8ad8d30",
                "Stu_0daa76a1a67ca9c9dc3a89d615b543c4",
                "Stu_c8a8f71fa26acc0ca1b7058cd6449d0d",
                "Stu_cb7abc51b57d959e65250b0c84e3b5c2",
                "Stu_77465918870d62f904c39f274daf2c13",
                "Stu_abf852c8786d9a932a5856b60e258b45",
                "Stu_a01f3fbbd697a9f53229969a6db3266f",
                "Stu_5e495eea98f0f8580e9cbfc5d8facc4f",
                "Test_20bcbf7e1daf2f6e6d57764460872a45",
                "Test_fc78cf61ec11fcae38da6dc5f6ef37b8",
                "Test_0027f6fd3c0cc6493aaf36c85912c6de",
                "Stu_000e6fce9a830b756649b290693c391c",
                "Stu_f77ef40b190d98062cdcdb672ba7d2e8",
                "Stu_e76a37825cd08e44aa0f4ab1521a1083",
                "Test_a35fd739a526f7feba6b301c22bd5500",
                "Test_2d6d533d6f351795bb56863908f5d49e",
                "Stu_67d8819b9ed4c75f6d40296a24e58248",
                "Stu_a61c8acfb718a0db2479512795cbd161",
                "Stu_a4771c7038ffd0de0e781e195641192d",
                "Stu_a279a116fa9348ae1c3ab2c085e89ea4",
                "Stu_3aa219c22d70529fb15761d3e4e29a56",
                "Stu_dce31fd7bc7eec7a36ab9850b5793570",
                "Test_921ebf77edce49efe06a17dc458efc26",
                "Stu_cd69d60c9584fbdd2bbc5b018143b52e",
                "Stu_637deeaa7b179dada45476d2ff61ca9f",
                "Stu_2382dac41fcbbbe44bef91620e013616",
                "Stu_0a73999b4fa456eb9389110ae74b1ca8",
                "Test_d65706f9bdcc67d0891b60c85bc121a5",
                "Stu_963189491820469a2de89e0c86c2a31d",
                "Stu_a51a21d3e328dd1489d8b2f2c645de99",
                "Stu_f49f2d7f606e3e49f7c6841d76ce7a14",
                "Stu_948b9e8a4be7d6a66d575922b8f2d19a"]


test_student_ids = ["Stu_e55159d745aec47df267867a6ec33fa4",
                    "Stu_a8e8c1b9475dbb03a2b0bb67f8934a26",
                    "Stu_5e5017ecca43ec6058e6138394711a6e",
                    "Stu_3c335140950e351d8698b6750995a1f7",
                    "Stu_a42ed68eaa9141843e4088d54a291fd9",
                    "Stu_476f17a89a733c2991c7cc4dbc7d92c5",
                    "Stu_2830644026610506220a0149cff8452c",
                    "Stu_b0727c282d9e5b19c6fed55887b74ce7",
                    "Stu_8793e1df399f67fb83db3270c2f06819",
                    "Stu_4bb7473597483fcfb001573b9f35a96b",
                    "Stu_cd72200df941618016cc9ff510c8792c",
                    "Stu_5260ecddc0a0bcd8abe778067936b758",
                    "Stu_dba11c31f0c4d71c83b574f8178b7393",
                    "Stu_334dadc097157485d273d8d7cab371e9",
                    "Stu_67212f1e68ed20cb040fd2f1c74f6bd7"]


#save the dev baseline set as a csv
dev_data = data_cleaned[data_cleaned['Anon Student Id'].isin(dev_student_ids)]
dev_data.to_csv("dev_data.csv",index=False)

#save the dev engineered set as a csv
dev_data_engineered = data_engineered[data_engineered['Anon Student Id'].isin(dev_student_ids)]
dev_data_engineered.to_csv("dev_data_engineered.csv",index=False)

#save the cv baseline set as a csv
cv_data = data_cleaned[data_cleaned['Anon Student Id'].isin(cv_student_ids)]
cv_data.to_csv("cv_data.csv",index=False)

#save the cv engineered set as a csv
cv_data_engineered = data_engineered[data_engineered['Anon Student Id'].isin(cv_student_ids)]
cv_data_engineered.to_csv("cv_data_engineered.csv",index=False)

#save the test engineered set as a csv
test_data_engineered = data_engineered[data_engineered['Anon Student Id'].isin(test_student_ids)]
test_data_engineered.to_csv("test_data_engineered.csv",index=False)
