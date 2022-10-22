# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def get_bool_from_input(question):
    answer = input('%s [y/n]?:'%question)
    if answer == 'y' or answer == 'yes':
        answer = True
    elif answer == 'n' or answer == 'no':
        answer = False
    else:
        print('Please enter y or n only!')
        return get_bool_from_input(question)
    return answer

def get_int_from_input(question):
    answer = input('%s [int]?:'%question)
    try:
       answer = int(answer)
    except ValueError:
       print("That's not an integer!")
       return get_int_from_input(question)
    return answer

def get_float_from_input(question):
    answer = input('%s [float]?:'%question)
    try:
       answer = float(answer)
    except ValueError:
       print("That's not a float number!")
       return get_float_from_input(question)
    return answer