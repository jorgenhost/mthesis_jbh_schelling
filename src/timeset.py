from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import overload, Optional, Set


@dataclass(frozen=True)
class ContinuousTimeRange:
    start: datetime
    end: datetime    
    
    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("Start cannot be greater than end.")
        
    @property
    def as_timedelta(self) -> timedelta:
        return self.end - self.start
    
    def __contains__(self, moment: datetime) -> bool:
        return self.start <= moment <= self.end
    

class TimeRange:

    @overload  # hint for the type checker 
    def __init__(self):
        pass

    @overload  # hint for the type checker
    def __init__(self, start: datetime, end: datetime):
        pass

    def __init__(self, start: Optional[datetime] = None, end: Optional[datetime] = None):
        self.periods: Set[ContinuousTimeRange] = set()
        if start and end:
            self.periods.add(ContinuousTimeRange(start, end))
        elif start or end:
            raise ValueError("A `TimeRange` must have either none or both `start` and `end` specified.")

    def __repr__(self) -> str:
        return ' + '.join(
            [f"TimeRange(start={repr(p.start)}, end={repr(p.end)})" for p in self.periods]
        ) or "TimeRange()"
    
    def __bool__(self) -> bool:
        return len(self._periods) != 0

    @property
    def as_timedelta(self) -> timedelta:
        return sum([p.as_timedelta for p in self._periods])
    
    def __contains__(self, moment: datetime) -> bool:
        return any([moment in p for p in self._periods])