'''
0-4 -> Minimal depression
5-9 -> Mild depression
10-14 -> Moderate depression
15-19 -> Moderately severe depression
20-24 -> Severe depression
'''
def place_value_in_bin(value, put_in_bin=False):
    if put_in_bin:
        if value <= 4:
            return 0
        elif value <= 9:
            return 1
        elif value <= 14:
            return 2
        elif value <= 19:
            return 3
        else:
            return 4
    else:
        return value