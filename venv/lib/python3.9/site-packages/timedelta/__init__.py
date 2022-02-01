__all__ = ["Total", "Timedelta"]


import datetime

"""
https://docs.python.org/3/library/datetime.html#timedelta-objects
"""


class Total:
    __readme__ = ["__init__", "seconds", "minutes", "hours",  "days"]
    total_seconds = None

    def __init__(self, total_seconds):
        """init from total seconds count"""
        self.total_seconds = total_seconds

    @property
    def seconds(self):
        """return total seconds count"""
        return int(self.total_seconds)

    @property
    def minutes(self):
        """return total minutes count"""
        return int(self.seconds / 60)

    @property
    def hours(self):
        """return total hours count"""
        return int(self.minutes / 60)

    @property
    def days(self):
        """return total days count"""
        return int(self.hours / 24)


class Timedelta(datetime.timedelta):
    """datetime.timedelta replacement"""

    def __new__(self, *args, **kwargs):
        if args:
            kwargs["seconds"] = args[0].total_seconds()
            kwargs["microseconds"] = args[0].microseconds
        return datetime.timedelta.__new__(self, **kwargs)

    @property
    def total(self):
        """return Total object"""
        return Total(self.total_seconds())
