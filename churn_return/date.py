#
# ps9pr1.py (Problem Set 9, Problem 1)
#
# A class to represent calendar dates
#
# name: Jordan Love
# email: jlove@bu.edu
#
# If you worked with a partner, put his or her contact info below:
# partner's name:
# partner's email:
#

class Date:
    """A class that stores and manipulates dates,
       represented by a day, month, and year.
    """

    # The constructor for the Date class.
    def __init__(self, new_month, new_day, new_year):
        """The constructor for objects of type Date."""
        self.month = new_month
        self.day = new_day
        self.year = new_year


    # The function for the Date class that returns a Date
    # object in a string representation.
    def __str__(self):
        """This method returns a string representation for the
           object of type Date that calls it (named self).

           ** Note that this _can_ be called explicitly, but
              it more often is used implicitly via printing.
        """
        s =  "%02d/%02d/%04d" % (self.month, self.day, self.year)
        return s

    def __hash__(self):
        string = ""
        string += str(self.month)
        string += str(self.day)
        string += str(self.year)
        return string.__hash__()

    def is_leap_year(self):
        """ Returns True if the calling object is
            in a leap year. Otherwise, returns False.
        """
        if self.year % 400 == 0:
            return True
        elif self.year % 100 == 0:
            return False
        elif self.year % 4 == 0:
            return True
        return False


    def copy(self):
        """ Returns a new object with the same month, day, year
            as the calling object (self).
        """
        new_date = Date(self.month, self.day, self.year)
        return new_date


    def tomorrow(self):
        """ changes the called object so that it represents one calendar day
            after the date that it originally represented.
        """

        days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.day += 1
        if self.is_leap_year():
            days_in_month[2] = 29

        if days_in_month[self.month] < self.day:
            self.day = 1
            self.month += 1

        if self.month > 12:
            self.month = 1
            self.year +=1

    def  add_n_days(self, n):
        """changes the calling object so that it represents n calendar days
           after the date it originally represented.
           input:n is a non-negative integer
        """

        for i in range(n):
            self.tomorrow()

    def __eq__(self, other):
        """returns True if the called object (self) and the argument (other)
           represent the same calendar date
        """
        if self.day == other.day and self.month == other.month and self.year == other.year:
            return True
        return False

    def  is_before(self, other):
        """returns True if the called object represents a calendar date that
           occurs before the calendar date that is represented by other.
           input: other is an object of the Date class
        """

        if self.year < other.year:
            return True
        elif self.year == other.year and self.month < other.month:
            return True
        elif self.year == other.year and self.month == other.month and self.day < other.day:
            return True
        return False

    def is_after(self, other):
        """returns True if the calling object represents a calendar date that
           occurs after the calendar date that is represented by other
           input: other is an object of the Date class
        """

        if self.year > other.year:
            return True
        elif self.year == other.year and self.month > other.month:
            return True
        elif self.year == other.year and self.month == other.month and self.day > other.day:
            return True
        return False


    def diff(self, other):
        """returns an integer that represents the number of days between self
           and other.
           input: other is an object of the Date class
        """

        difference = 0
        first = self.copy()
        second = other.copy()

        if first.is_after(second):
            while True:
                if first == second:
                    break
                second.tomorrow()
                difference += 1

        else:
            while True:
                if first == second:
                    break
                first.tomorrow()
                difference -= 1

        return difference


    def  day_of_week(self):
        """returns a string that indicates the day of the week of the Date
           object that calls it.
        """

        day_of_week_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                     'Friday', 'Saturday', 'Sunday']

        relative = Date(11, 16, 2014)


        return day_of_week_names[self.diff(relative)%7 - 1]
