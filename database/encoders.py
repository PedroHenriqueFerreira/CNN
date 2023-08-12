import json

from typing import Any

from database.database import Database    

class Encoder:
    ''' Abstract encoder class '''
    
    def fit(self, database: Database) -> None:
        ''' Fit the encoder to the data '''
        
        raise NotImplementedError()
    
    def transform(self, database: Database) -> Database:
        ''' Transform the data using the encoder '''
        
        raise NotImplementedError()    
    
    def inverse_transform(self, database: Database) -> Database:
        ''' Inverse transform the data using the encoder '''
        
        raise NotImplementedError()
    
    def fit_transform(self, database: Database) -> Database: 
        ''' Fit the encoder to the data and transform the data using the encoder '''
        
        self.fit(database)
        
        return self.transform(database)
    
    def to_json(self, path: str) -> None:
        ''' Save the encoder to a JSON file '''
        
        attributes = self.__dict__
        
        with open(path, 'w') as file:
            json.dump(attributes, file)
          
    @staticmethod  
    def from_json(path: str) -> 'Encoder': 
        ''' Load the encoder from a JSON file '''
        
        raise NotImplementedError()

class LabelEncoder(Encoder):
    ''' Label encode the data '''
    
    def __init__(self):
        self.categories: list[list[Any]] = []
        
    def fit(self, database: Database) -> None:
        self.categories.clear()
        
        for column in database.columns:
            self.categories.append(database[column].unique())
    
    def transform(self, database: Database) -> Database:
        if len(self.categories) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if len(self.categories) != len(database.columns):
            raise ValueError('Invalid database')
        
        columns = database.columns[:]
        values: list[list[Any]] = []
        
        for row in database.values:
            values.append([self.categories[i].index(item) for i, item in enumerate(row)])
        
        return Database(columns, values)
    
    def inverse_transform(self, database: Database) -> Database:
        if len(self.categories) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if len(self.categories) != len(database.columns):
            raise ValueError('Invalid database')
    
        columns = database.columns[:]
        values: list[list[Any]] = []
        
        for row in database.values:
            values.append([self.categories[i][item] for i, item in enumerate(row)])
    
        return Database(columns, values)
    
    @staticmethod
    def from_json(path: str) -> 'LabelEncoder':        
        with open(path, 'r') as file:
            attributes = json.loads(file.read())
        
        encoder = LabelEncoder()
        encoder.__dict__.update(attributes)
        
        return encoder
    
class OneHotEncoder(Encoder):
    ''' One-hot encode the data '''
    
    def __init__(self):
        self.columns: list[str] = []
        self.categories: list[list[Any]] = []
        
    def fit(self, database: Database) -> None:
        self.columns.clear()
        self.categories.clear()
        
        for column in database.columns:
            self.columns.append(column)
            self.categories.append(database[column].unique())
    
    def transform(self, database: Database) -> Database:
        if len(self.categories) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if len(self.categories) != len(database.columns):
            raise ValueError('Invalid database')
        
        columns: list[str] = []
        
        for column, category in zip(database.columns, self.categories):
            columns.extend([f'{column}_{value}' for value in category])
        
        values: list[list[Any]] = []
        
        for row in database.values:
            values.append([])
            
            for item, category in zip(row, self.categories):
                values[-1].extend([int(item == value) for value in category])
        
        return Database(columns, values)
    
    def inverse_transform(self, database: Database) -> Database:
        if len(self.categories) == 0:
            raise ValueError('Encoder must be fitted before transforming data')
        
        if sum(len(category) for category in self.categories) != len(database.columns):
            raise ValueError('Invalid database')
        
        columns = self.columns[:]
        values: list[list[Any]] = []
        
        for row in database.values:
            values.append([])
            
            index = 0
            
            for category in self.categories:
                values[-1].append(category[row[index : index + len(category)].index(1)])
                
                index += len(category)
        
        return Database(columns, values)
    
    @staticmethod
    def from_json(path: str) -> 'OneHotEncoder':
        with open(path, 'r') as file:
            attributes = json.loads(file.read())
        
        encoder = OneHotEncoder()
        encoder.__dict__.update(attributes)
        
        return encoder
    