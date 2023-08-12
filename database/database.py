from typing import Any, Callable, Union

class Database:
    ''' Data base class '''
    
    def __init__(self, columns: list[str] | None = None, values: list[list[Any]] | None = None):
        ''' Create a new data base instance with the given columns and values '''
        
        if values is None:
            values = []
        
        if columns is None:
            columns = []
            
            if len(values) > 0: 
                columns.extend([str(i) for i in range(len(values[0]))])
            
        if values is None:
            values = []
        
        if len(set(columns)) != len(columns):
            raise ValueError('Columns must be unique')
        
        if len(values) > 0 and len(columns) != len(values[0]):
            raise ValueError('Columns and values are not compatible')
        
        if len(set([len(row) for row in values])) > 1:
            raise ValueError('Invalid values shape')
        
        self.columns = columns
        self.values = values
    
    def __str__(self):
        ''' Return a string representation of the data base '''
        
        columns_width: list[int] = []
        
        for i in range(len(self.columns)):
            width = max([len(str(row[i])) for row in [self.columns, *self.values]])
            
            columns_width.append(width)
        
        lines: list[str] = []
        
        for row in [self.columns, *self.values]:
            lines.append('  '.join([str(item).rjust(columns_width[i]) for i, item in enumerate(row)]))
            
        return '\n'.join(lines)

    def map(self, func: Callable[[Any], Any]) -> 'Database':
        ''' Map the data base with the given function '''
        
        return Database(self.columns[:], [[func(item) for item in row] for row in self.values])

    def __invert__(self) -> 'Database':
        ''' Return the inverse of the data base values '''
        
        return self.map(lambda x: not x)

    def __gt__(self, other: int | float) -> 'Database':
        ''' Return if the data base values is greater than the given value'''
        
        return self.map(lambda x: x > other)

    def __lt__(self, other: int | float) -> 'Database':
        ''' Return if the data base values is less than the given value '''
        
        return self.map(lambda x: x < other)
    
    def __ge__(self, other: int | float) -> 'Database':
        ''' Return if the data base values is greater than or equal to the given value '''
        
        return self.map(lambda x: x >= other)
    
    def __le__(self, other: int | float) -> 'Database':
        ''' Return if the data base values is less than or equal to the given value '''
        
        return self.map(lambda x: x <= other)
    
    def __eq__(self, other: int | float) -> 'Database': # type: ignore
        ''' Return if the data base values is equal to the given value '''
        
        return self.map(lambda x: x == other)
    
    def __ne__(self, other: int | float) -> 'Database':  # type: ignore
        ''' Return if the data base values is not equal to the given value '''
        
        return self.map(lambda x: x != other)

    def is_in(self, other: list[Any]) -> 'Database':
        ''' Return if the data base values is in the given list '''
        
        return self.map(lambda x: x in other)

    def __and__(self, other: 'Database') -> 'Database':
        ''' Return the & operator between the data base and the given data base '''

        if len(self.values) != len(other.values):
            raise ValueError('Invalid data base shape') 
        
        values = [[all(row) & all(other_row)] for row, other_row in zip(self.values, other.values)]
        
        return Database(['and'], values)
    
    def __or__(self, other: 'Database') -> 'Database':
        ''' Return the | operator between the data base and the given data base '''

        if len(self.values) != len(other.values):
            raise ValueError('Invalid data base shape') 
        
        values = [[all(row) | all(other_row)] for row, other_row in zip(self.values, other.values)]
        
        return Database(['or'], values)

    def __xor__(self, other: 'Database') -> 'Database':
        ''' Return the ^ operator between the data base and the given data base '''

        if len(self.values) != len(other.values):
            raise ValueError('Invalid data base shape') 
        
        values = [[all(row) ^ all(other_row)] for row, other_row in zip(self.values, other.values)]
        
        return Database(['xor'], values)

    def __getitem__(
        self, 
        key: Union[str, int, slice, list[str | int | slice], tuple[str | int | slice, ...], 'Database']
    ) -> 'Database':
        ''' Filter the data base with the given key '''
        
        indexes: list[int] = []
        
        key = key if isinstance(key, list | tuple | Database) else [key]
        
        if isinstance(key, list | tuple):
            for column in key:
                if isinstance(column, int):
                    if column < -len(self.columns) or column >= len(self.columns):
                        raise ValueError(f'Column index {column} out of range')
                    
                    column = column if column >= 0 else len(self.columns) + column
                    
                    indexes.append(column)
                    
                elif isinstance(column, str):
                    if column not in self.columns:
                        raise ValueError(f'Column {column} not found')
                    
                    indexes.append(self.columns.index(column))
                
                elif isinstance(column, slice):
                    slice_values = [column.start, column.stop, column.step]
            
                    if isinstance(slice_values[-1], str):
                        raise ValueError(f'Invalid step {slice_values[-1]}')
                    
                    for i, slice_value in enumerate(slice_values):
                        if isinstance(slice_value, str):
                            if slice_value not in self.columns:
                                raise ValueError(f'Column {slice_value} not found')
                            
                            slice_values[i] = self.columns.index(slice_value)
                        elif not isinstance(slice_value, int) and slice_value is not None:
                            raise ValueError(f'Invalid index {slice_value}')
                    
                    range_values = slice(*slice_values).indices(len(self.columns))
                    
                    indexes.extend(list(range(*range_values)))
                
                else:
                    raise ValueError(f'Invalid column {column}')
                
            columns = [self.columns[index] for index in indexes]
            values = [[row[index] for index in indexes] for row in self.values]
        
        elif isinstance(key, Database):
            if len(self.values) != len(key.values):
                raise ValueError('Invalid data base shape')
            
            for column in key.columns:
                if column in self.columns + ['and', 'or', 'xor']:
                    continue
                
                raise ValueError(f'Invalid data base columns')
            
            columns = self.columns[:]
            values = [row[:] for row, other_row in zip(self.values, key.values) if all(other_row)]
        
        else:
            raise ValueError(f'Invalid key {key}')
    
        return Database(columns, values)
    
    def __setitem__(
        self,
        key: str | int | slice | list[str | int | slice] | tuple[str | int | slice, ...], 
        values: Union[list[Any], tuple[Any, ...], 'Database']
    ) -> None:
        ''' Set the given column with the given values '''
        
        key = key if isinstance(key, list | tuple) else [key]
        
        if len(key) > 1 and isinstance(values, Database):
            raise ValueError(f'Cannot set multiple columns with a data base')
        
        indexes: list[int] = []
        columns: list[str] = []

        if isinstance(key, list | tuple):
            new_index = len(self.columns)
            
            for column in key:
                if isinstance(column, int):
                    if column < -len(self.columns) or column >= len(self.columns):
                        raise ValueError(f'Column index {column} out of range')
                    
                    column = column if column >= 0 else len(self.columns) + column
                    
                    indexes.append(column)
                    columns.append(self.columns[column])
                    
                elif isinstance(column, str):
                    if column not in self.columns:
                        if isinstance(values, Database):
                            raise ValueError(f'Column {column} not found')
                        else:
                            indexes.append(new_index)
                            new_index += 1
                    else:
                        indexes.append(self.columns.index(column))
                    
                    columns.append(column)
                
                elif isinstance(column, slice):
                    slice_values = [column.start, column.stop, column.step]

                    step = slice_values[-1]
            
                    if isinstance(step, str):
                        raise ValueError(f'Invalid step {slice_values[-1]}')
                    
                    for i, slice_value in enumerate(slice_values):
                        if isinstance(slice_value, str):
                            if slice_value not in self.columns:
                                raise ValueError(f'Column {slice_value} not found')
                            
                            slice_values[i] = self.columns.index(slice_value)
                        elif not isinstance(slice_value, int) and slice_value is not None:
                            raise ValueError(f'Invalid index {slice_value}')
                    
                    range_values = slice(*slice_values).indices(len(self.columns))
                    
                    indexes.extend(list(range(*range_values)))
                    columns.extend([self.columns[i] for i in range(*range_values)])    
                
                else:
                    raise ValueError(f'Invalid column {column}')
                
        else:
            raise ValueError(f'Invalid key {key}')
    
        if len(set(columns)) != len(columns):
            raise ValueError('Columns must be unique')
    
        del self[[index for index in indexes if index < len(self.columns)]]
        
        if isinstance(values, Database):
            indexes = list(range(indexes[0], indexes[0] + len(values.columns)))
        
        columns = values.columns if isinstance(values, Database) else columns
        values = values.values if isinstance(values, Database) else values
        
        values = [value if isinstance(value, list | tuple) else [value] for value in values]
        
        if len(self.values) > 0 and len(self.values) != len(values):
            raise ValueError('Invalid values shape')
        
        if len(values) > 0 and len(values[0]) != len(columns):
            raise ValueError('Invalid values shape')
        
        columns = [column for _, column in sorted(zip(indexes, columns))]
        values = [[item for _, item in sorted(zip(indexes, value))] for value in values]
        
        indexes = sorted(indexes)
        
        for i, column in enumerate(columns):
            if column in self.columns:
                raise ValueError(f'Column {column} already exists')
            
            self.columns.insert(indexes[i], column)
            
            if len(self.values) == 0:
                for other_row in values:
                    self.values.append([other_row[i]])
            else:
                for row, other_row in zip(self.values, values):
                    row.insert(indexes[i], other_row[i])

    def __delitem__(
        self, 
        key: str | int | slice | list[str | int | slice] | tuple[str | int | slice, ...]
    ) -> None:
        ''' Delete the given column '''
        
        columns: list[str] = []
        
        key = key if isinstance(key, list | tuple) else [key]

        if isinstance(key, list | tuple):
            for column in key:
                if isinstance(column, int):
                    if column < -len(self.columns) or column >= len(self.columns):
                        raise ValueError(f'Column index {column} out of range')
                    
                    column = column if column >= 0 else len(self.columns) + column
                    
                    columns.append(self.columns[column])
                    
                elif isinstance(column, str):
                    if column not in self.columns:
                        raise ValueError(f'Column {column} not found')
                    
                    columns.append(column)
                
                elif isinstance(column, slice):
                    slice_values = [column.start, column.stop, column.step]
            
                    if isinstance(slice_values[-1], str):
                        raise ValueError(f'Invalid step {slice_values[-1]}')
                    
                    for i, slice_value in enumerate(slice_values):
                        if isinstance(slice_value, str):
                            if slice_value not in self.columns:
                                raise ValueError(f'Column {slice_value} not found')
                            
                            slice_values[i] = self.columns.index(slice_value)
                        elif not isinstance(slice_value, int) and slice_value is not None:
                            raise ValueError(f'Invalid index {slice_value}')
                    
                    range_values = slice(*slice_values).indices(len(self.columns))
                    
                    columns.extend([self.columns[i] for i in range(*range_values)])
                
                else:
                    raise ValueError(f'Invalid column {column}')
                            
        else:
            raise ValueError(f'Invalid key {key}')
        
        if len(set(columns)) != len(columns):
            raise ValueError('Columns must be unique')
        
        for column in columns:
            index = self.columns.index(column)
            
            del self.columns[index]
            
            for row in self.values:
                del row[index]
            
    def unique(self) -> list[Any]:
        ''' Return the unique values of the data base '''
        
        values: list[Any] = []
        
        for row in self.values:
            for item in row:
                if item in values:
                    continue
                
                values.append(item)
        
        return values
    
    def sum(self) -> float:
        ''' Return the sum of all items in the data base '''
        
        result = 0.0
        
        for row in self.values:
            for item in row:
                if not isinstance(item, int | float | bool):
                    raise ValueError('Data base must contain only numbers or booleans for sum')
                
                result += item
        
        return result
    
    def count(self) -> int:
        ''' Return the number of items in the data base '''
        
        return len(self.values) * len(self.columns)
    
    def mean(self) -> float:
        ''' Return the mean of all values in the data base '''
        
        if self.count() == 0:
            return 0
        
        return self.sum() / self.count()
    
    def std(self) -> float:
        ''' Return the standard deviation of all values in the data base '''
        
        if self.count() == 0:
            return 0
        
        mean = self.mean()   
             
        var = self.map(lambda item: (item - mean) ** 2).sum() / self.count()
        
        std: float = var ** 0.5
        
        return std
    
    def split(self, ratio: float = 0.8) -> tuple['Database', 'Database']:
        ''' Split the data base into two data bases by the given ratio '''  
        
        if ratio <= 0 or ratio >= 1:
            raise ValueError('Invalid ratio')
        
        columns = self.columns
        values = [row[:] for row in self.values]
        index = int(len(values) * ratio)
        
        return Database(columns[:], values[:index]), Database(columns[:], values[index:])
    
    @staticmethod
    def from_csv(path: str, separator: str = ',', columns: list[str] | None = []) -> 'Database':
        ''' Read a csv file and return a data base instance '''

        values: list[list[Any]] = []
        
        with open(path, 'r') as file:
            for line in file.readlines():
                items = line.strip().split(separator)

                row: list[Any] = []                

                for item in items:
                    item = item.strip()
                    
                    if item.isnumeric():
                        row.append(int(item))
                    elif item.replace('.', '', 1).isnumeric():
                        row.append(float(item))
                    else:
                        row.append(item)
                    
                if columns is not None and len(columns) == 0:
                    columns.extend([str(item) for item in row])
                else:
                    values.append(row)
            
        return Database(columns, values)
    
    def to_csv(self, path: str, separator: str = ',', columns: list[str] | None = []) -> None:
        ''' Save the data base to a csv file '''
        
        with open(path, 'w') as file:
            if columns is not None:
                columns = columns if len(columns) > 0 else self.columns
            
            for row in [columns] + self.values:
                if row is None:
                    continue
                    
                file.write(separator.join([str(item) for item in row]) + '\n')

__all__ = ['Database']