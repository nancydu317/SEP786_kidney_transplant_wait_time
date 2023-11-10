# Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# Settings
path_to_csv = "../waitlist_kidney_brazil.csv"
columns_to_remove = ["Id", 
                     "date", 
                    "age_cat",
                    "cPRA_cat",
                    "calculated_frequency_DR.f1",
                    "calculated_frequency_DR.f2",
                    "calculated_frequency_DR.f",
                    "calculated_frequency_B.f1",
                    "calculated_frequency_B.f2",
                    "calculated_frequency_B.f",
                    "calculated_frequency_A.f1",
                    "calculated_frequency_A.f2",
                    "calculated_frequency_A.f",
                    "date_acutal",
                    "Time_death",
                    "Transplant_Y_N",
                    "X36MthsTx",
                    "priorization",
                    "razon_removed",
                    "Time_Tx",
                    "patient_still_on_list",
                    "Transplant",
                    "removed_list",
                    "death"]

port_to_english = {
    'Não' :False, 
    'Sim' :True, 
    'Maior.60' :'>60', 
    '18.a.60' :'Between_18_60', 
    'Menor.18' :'<18', 
    'Branca' :'White', 
    'Parda' :'Brown', 
    'Negra' :'Black', 
    'Amarela' :'Yellow', 
    'Entre_50_80' :'Between_50_80', 
    'Entre_0_50' :'Between_0_50', 
    'Maior_80' :'More_80', 
    'Óbito Lista' :'Death_list', 
    'heterozigoto' :'heterozygous', 
    'homozigoto' :'homozygous', 
    'Outras' : 'Others'
}

def encode_binary_columns(dataframe):
    """Encode all binaries columns."""
    le = LabelEncoder()
    for column in dataframe.columns:
        if dataframe[column].nunique() == 2:
            dataframe[column] = le.fit_transform(dataframe[column])
    return dataframe

def one_hot_encode(dataframe):
    """ Function to apply one-hot encoding to categorical columns."""
    categorical_columns = dataframe.select_dtypes(include=['object']).columns
    dataframe = pd.get_dummies(dataframe, columns=categorical_columns)
    return dataframe

class GetData:
    """A simple example class"""
    def __init__(self, path = path_to_csv):
        self.raw_data = pd.read_csv(path)
        self.cleaned_data = self.get_cleaned_data()
    
    def get_cleaned_data(self, 
                     columns_remove = columns_to_remove, 
                     dict_portuguese_english = port_to_english,
                     removenan = True):
        """Remove uselles columns, translate portuguese to English."""
        cleaned_data = self.raw_data.copy()

        #Translating Portuguese to English
        for portuguese,english in dict_portuguese_english.items():
            cleaned_data.replace(to_replace= portuguese, value= english, inplace=True)

        # Keeping People that did got a surgery and people who did not but are still alive and on the waitlist.
        cleaned_data = cleaned_data[(cleaned_data['Transplant'] == True) | ((cleaned_data['death'] == False) & (cleaned_data['event'] == 0))]

        # Dropping useless columns.
        cleaned_data = cleaned_data.drop(columns_remove, axis=1)

        # Fill Nan valus for number_gestation.
        cleaned_data['number_gestation'] = cleaned_data['number_gestation'].fillna(0)

        if removenan == True:
            return cleaned_data.dropna()
        return cleaned_data
    
    def get_splitted_scale_encoded_data(self, outcome = 'time'):
        #Encode binary
        tempdf = encode_binary_columns(self.cleaned_data)

        #one_hot_encode for categorical values
        tempdf = one_hot_encode(tempdf)

        #Get rid of redundant diabetes
        tempdf = tempdf.drop("underline_disease_Diabetes", axis=1)
        
        #Splitting test and train
        traindata = tempdf[tempdf['event'] == 1]
        testdata = tempdf[tempdf['event'] == 0]

        #Getting Y_train,XTrain, ...
        Y_train = traindata[outcome]
        X_train = traindata.drop([outcome,"event"],axis=1) 

        Y_test = testdata[outcome]
        X_test = testdata.drop([outcome,"event"],axis=1) 

        #Scaling values
        sc = MinMaxScaler()
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.transform(X_test)
        
        return X_train_scaled, Y_train, X_test_scaled, Y_test
    

if __name__ == '__main__':
    Data = GetData()
    print(Data.get_splitted_scale_encoded_data())