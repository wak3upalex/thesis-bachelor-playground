# dataframe
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}

df = pd.DataFrame(data=data)
print(df)

# Pydantic
from pydantic import BaseModel, ValidationError
from typing import Optional
from datetime import date


class User(BaseModel):
    id: int
    name: str
    signup_ts: Optional[date] = None
    friends: list[int] = []


input_data = {
    "id": "123",
    "name": "John Doe",
    "signup_ts": "2025-06-01",
    "friends": [1, 2, 3]
}

try:
    user = User(**input_data)
    print(user)
except ValidationError as e:
    print(e)




# Пример работы с datasets
from datasets import load_dataset

dataset = load_dataset( "json", data_files="tests_datasets/my_data.jsonl", split="train")

print("Size of dataset:", len(dataset))

first_record = dataset[0]
print("First record:", first_record)
print("Full dataset \n", dataset)

