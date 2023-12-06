from dataclasses import dataclass
from typing import Union


@dataclass
class Transaction:
    merchant: int
    category: int
    amt: float
    city: int
    state: int
    lat: float
    long: float
    city_pop: int
    job: int
    merch_lat: float
    merch_long: float
    age: int
    is_fraud: Union[bool, None] = None

    @classmethod
    def from_object(cls, obj):
        return cls(
            merchant=int(obj["merchant"]),
            category=int(obj["category"]),
            amt=float(obj["amt"]),
            city=int(obj["city"]),
            state=int(obj["state"]),
            lat=float(obj["lat"]),
            long=float(obj["long"]),
            city_pop=int(obj["city_pop"]),
            job=int(obj["job"]),
            merch_lat=float(obj["merch_lat"]),
            merch_long=float(obj["merch_long"]),
            age=int(obj["age"]),
        )

    def __repr__(self):
        return (f"Transaction(merchant={self.merchant}, amt={self.amt}, city={self.city}, "
                f"state={self.state}, is_fraud={self.is_fraud})")
