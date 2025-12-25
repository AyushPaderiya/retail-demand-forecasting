"""
Calendar Data Generator

Generates calendar dimension with date features and holidays.
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd

from src.data_generation.base_generator import BaseGenerator
from src.data_generation.config import DataGenerationConfig
from src.exception import DataGenerationException


class CalendarGenerator(BaseGenerator):
    """Generator for calendar/date dimension data."""
    
    # US Federal Holidays (simplified - fixed dates)
    HOLIDAYS = {
        (1, 1): "New Year's Day",
        (7, 4): "Independence Day",
        (11, 11): "Veterans Day",
        (12, 25): "Christmas Day",
        (12, 31): "New Year's Eve",
    }
    
    # Floating holidays (approximate dates)
    FLOATING_HOLIDAYS = [
        "Martin Luther King Jr. Day",  # 3rd Monday of January
        "Presidents' Day",              # 3rd Monday of February
        "Memorial Day",                 # Last Monday of May
        "Labor Day",                    # 1st Monday of September
        "Columbus Day",                 # 2nd Monday of October
        "Thanksgiving",                 # 4th Thursday of November
    ]
    
    @property
    def filename(self) -> str:
        return "calendar.csv"
    
    def generate(self) -> pd.DataFrame:
        """
        Generate calendar data.
        
        Returns:
            DataFrame with date dimension information.
        """
        try:
            start_date, end_date = self.config.get_date_range()
            num_days = self.config.get_num_days()
            
            self.logger.info(
                f"Generating calendar from {start_date} to {end_date} "
                f"({num_days} days)"
            )
            
            dates = []
            current_date = start_date
            
            while current_date <= end_date:
                is_holiday, holiday_name = self._check_holiday(current_date)
                
                # Calculate days to next major holiday
                days_to_holiday = self._days_to_next_holiday(current_date)
                
                # Determine season
                season = self._get_season(current_date)
                
                # Fiscal quarter (assuming fiscal year = calendar year)
                fiscal_quarter = (current_date.month - 1) // 3 + 1
                
                dates.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "year": current_date.year,
                    "month": current_date.month,
                    "day": current_date.day,
                    "day_of_week": current_date.weekday(),  # 0=Monday, 6=Sunday
                    "day_name": current_date.strftime("%A"),
                    "week_of_year": current_date.isocalendar()[1],
                    "quarter": (current_date.month - 1) // 3 + 1,
                    "fiscal_quarter": fiscal_quarter,
                    "is_weekend": current_date.weekday() >= 5,
                    "is_holiday": is_holiday,
                    "holiday_name": holiday_name,
                    "is_month_start": current_date.day == 1,
                    "is_month_end": self._is_month_end(current_date),
                    "days_to_holiday": days_to_holiday,
                    "season": season,
                })
                
                current_date += timedelta(days=1)
            
            df = pd.DataFrame(dates)
            
            holidays_count = df["is_holiday"].sum()
            weekends_count = df["is_weekend"].sum()
            
            self.logger.info(
                f"Generated {len(df)} calendar days with "
                f"{holidays_count} holidays and {weekends_count} weekend days"
            )
            
            return df
            
        except Exception as e:
            raise DataGenerationException(f"Failed to generate calendar: {e}", sys)
    
    def _check_holiday(self, date: datetime) -> Tuple[bool, str]:
        """Check if date is a holiday."""
        # Check fixed holidays
        key = (date.month, date.day)
        if key in self.HOLIDAYS:
            return True, self.HOLIDAYS[key]
        
        # Check floating holidays
        floating = self._get_floating_holidays(date.year)
        for holiday_date, name in floating:
            if date == holiday_date:
                return True, name
        
        return False, None
    
    def _get_floating_holidays(self, year: int) -> List[Tuple[datetime, str]]:
        """Calculate floating holidays for a given year."""
        holidays = []
        
        # MLK Day - 3rd Monday of January
        jan1 = datetime(year, 1, 1).date()
        mlk = self._nth_weekday(year, 1, 0, 3)  # 3rd Monday
        holidays.append((mlk, "Martin Luther King Jr. Day"))
        
        # Presidents' Day - 3rd Monday of February
        presidents = self._nth_weekday(year, 2, 0, 3)
        holidays.append((presidents, "Presidents' Day"))
        
        # Memorial Day - Last Monday of May
        memorial = self._last_weekday(year, 5, 0)
        holidays.append((memorial, "Memorial Day"))
        
        # Labor Day - 1st Monday of September
        labor = self._nth_weekday(year, 9, 0, 1)
        holidays.append((labor, "Labor Day"))
        
        # Columbus Day - 2nd Monday of October
        columbus = self._nth_weekday(year, 10, 0, 2)
        holidays.append((columbus, "Columbus Day"))
        
        # Thanksgiving - 4th Thursday of November
        thanksgiving = self._nth_weekday(year, 11, 3, 4)  # Thursday = 3
        holidays.append((thanksgiving, "Thanksgiving"))
        
        # Black Friday - Day after Thanksgiving
        black_friday = thanksgiving + timedelta(days=1)
        holidays.append((black_friday, "Black Friday"))
        
        return holidays
    
    def _nth_weekday(
        self, 
        year: int, 
        month: int, 
        weekday: int, 
        n: int
    ) -> datetime:
        """Find the nth occurrence of a weekday in a month."""
        from datetime import date
        first_day = date(year, month, 1)
        first_weekday = first_day.weekday()
        
        # Days until first occurrence of weekday
        days_until = (weekday - first_weekday) % 7
        
        # Date of nth occurrence
        day = 1 + days_until + (n - 1) * 7
        return date(year, month, day)
    
    def _last_weekday(self, year: int, month: int, weekday: int) -> datetime:
        """Find the last occurrence of a weekday in a month."""
        from datetime import date
        from calendar import monthrange
        
        last_day = monthrange(year, month)[1]
        last_date = date(year, month, last_day)
        
        # Days back to last occurrence
        days_back = (last_date.weekday() - weekday) % 7
        return date(year, month, last_day - days_back)
    
    def _days_to_next_holiday(self, current_date: datetime) -> int:
        """Calculate days until next major holiday."""
        major_holidays = [
            (12, 25),  # Christmas
            (11, 27),  # Approximate Thanksgiving
            (7, 4),    # Independence Day
            (1, 1),    # New Year
        ]
        
        min_days = 365
        current_year = current_date.year
        
        for month, day in major_holidays:
            try:
                holiday = datetime(current_year, month, day).date()
                if holiday <= current_date:
                    holiday = datetime(current_year + 1, month, day).date()
                
                days = (holiday - current_date).days
                min_days = min(min_days, days)
            except ValueError:
                continue
        
        return min_days
    
    def _get_season(self, date: datetime) -> str:
        """Determine season based on date."""
        month = date.month
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"
    
    def _is_month_end(self, date: datetime) -> bool:
        """Check if date is end of month."""
        next_day = date + timedelta(days=1)
        return next_day.month != date.month


if __name__ == "__main__":
    generator = CalendarGenerator()
    generator.run()
