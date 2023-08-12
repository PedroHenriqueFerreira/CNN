import json

from typing import Any

from database.database import Database

class Scaler:
    ''' Abstract scaler class '''
    
    def fit(self, database: Database) -> None:
        ''' Fit the scaler to the data '''
        
        raise NotImplementedError()
    
    def transform(self, database: Database) -> Database: 
        ''' Transform the data using the scaler '''
        
        raise NotImplementedError()
    
    def inverse_transform(self, database: Database) -> Database: 
        ''' Inverse transform the data using the scaler '''
        
        raise NotImplementedError()
    
    def fit_transform(self, database: Database) -> Database: 
        ''' Fit the scaler to the data and transform the data using the scaler '''
        
        self.fit(database)
    
        return self.transform(database)
    
    def to_json(self, path: str) -> None:
        ''' Save the encoder to a JSON file '''
        
        attributes = self.__dict__
        
        with open(path, 'w') as file:
            json.dump(attributes, file)
        
    @staticmethod
    def from_json(path: str) -> 'Scaler': 
        ''' Load the encoder from a JSON file '''
        
        raise NotImplementedError()

class NormalScaler(Scaler):
    ''' Normalize the data using the L2 norm '''
    
    def __init__(self):
        self.norms: list[float] = []
    
    def fit(self, database: Database) -> None:
        if database.map(lambda x: not isinstance(x, int | float)).sum() > 0:
            raise ValueError('Invalid database')
        
        self.norms.clear()
        
        for row in database.values:
            norm = sum(item ** 2 for item in row) ** 0.5
            
            self.norms.append(norm or 1)

    def transform(self, database: Database) -> Database:
        if len(self.norms) == 0:
            raise ValueError('Normalizer must be fitted before transforming data')
        
        if len(self.norms) != len(database.values):
            raise ValueError('Invalid database')
        
        columns = database.columns[:]
        values: list[list[Any]] = []
        
        for i, row in enumerate(database.values):
            values.append([item / self.norms[i] for item in row])
            
        return Database(columns, values)
    
    def inverse_transform(self, database: Database) -> Database:
        if len(self.norms) == 0:
            raise ValueError('Normalizer must be fitted before transforming data')
        
        if len(self.norms) != len(database.values):
            raise ValueError('Invalid database')
        
        columns = database.columns[:]
        values: list[list[Any]] = []
        
        for i, row in enumerate(database.values):
            values.append([item * self.norms[i] for item in row])
            
        return Database(columns, values)
    
    @staticmethod
    def from_json(path: str) -> 'NormalScaler':
        with open(path, 'r') as file:
            attributes = json.loads(file.read())
        
        encoder = NormalScaler()
        encoder.__dict__.update(attributes)
        
        return encoder

class StandardScaler(Scaler):
    ''' Standardize the data using the mean and standard deviation '''
    
    def __init__(self):
        self.mean: list[float] = []
        self.std: list[float] = []    
    
    def fit(self, database: Database) -> None:
        if database.map(lambda x: not isinstance(x, int | float)).sum() > 0:
            raise ValueError('Invalid database')
        
        self.mean.clear()
        self.std.clear()   
        
        for column in database.columns:
            self.mean.append(database[column].mean())
            self.std.append(database[column].std() or 1)
            
    def transform(self, database: Database) -> Database:
        if len(self.mean) == 0 or len(self.std) == 0:
            raise ValueError('Scaler must be fitted before transforming data')
        
        if len(self.mean) != len(database.columns) or len(self.std) != len(database.columns):
            raise ValueError('Invalid database')
        
        columns = database.columns[:]
        values: list[list[Any]] = []
        
        for row in database.values:
            values.append([(item - self.mean[i]) / self.std[i] for i, item in enumerate(row)])
            
        return Database(columns, values)
    
    def inverse_transform(self, database: Database) -> Database:
        if len(self.mean) == 0 or len(self.std) == 0:
            raise ValueError('Scaler must be fitted before transforming data')
        
        if len(self.mean) != len(database.columns) or len(self.std) != len(database.columns):
            raise ValueError('Invalid database')
        
        columns = database.columns[:]
        values: list[list[Any]] = []
        
        for row in database.values:
            values.append([(item * self.std[i]) + self.mean[i] for i, item in enumerate(row)])
        
        return Database(columns, values)
    
    @staticmethod
    def from_json(path: str) -> 'StandardScaler':
        with open(path, 'r') as file:
            attributes = json.loads(file.read())
        
        encoder = StandardScaler()
        encoder.__dict__.update(attributes)
        
        return encoder
    