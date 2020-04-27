import pandas as pd

##################
# AND PERCEPTRON #
##################

def and_perceptron(weight1, weight2, bias):
    # inputs and outputs
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    correct_outputs = [False, False, False, True]
    outputs = []
    is_correct= []

    # generate and ouput
    for test_input, correct_output in zip(test_inputs, correct_outputs):

        # linear algebra (Wx + b = y) - this is the percepton's logic
        linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias

        # everything below is nice formatting
        output = int(linear_combination >= 0)
        # checking if the output of this perceptron is acheiving the output needed to follow the logic
        is_correct_string = 'Yes' if output == correct_output else 'No'
        is_correct.append(is_correct_string)
        outputs.append([test_input[0], test_input[1], linear_combination, output])

    # review output
    num_wrong = len([is_correct for output in is_correct if is_correct == 'No'])
    output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Output'])
    print(output_frame)
    if not num_wrong:
        print('AND Perceptron Weighted well')


#################
# OR PERCEPTRON #
#################

def or_perceptron(weight1, weight2, bias):
    # inputs and outputs
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    correct_outputs = [False, True, True, True]
    outputs = []
    is_correct= []

    # generate and ouput
    for test_input, correct_output in zip(test_inputs, correct_outputs):

        # linear algebra (Wx + b = y) - this is the perceptron's logic
        linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias

        # everything below is nice formatting
        output = int(linear_combination >= 0)
        # checking if the output of this perceptron is acheiving the output needed to follow the logic
        is_correct_string = 'Yes' if output == correct_output else 'No'
        is_correct.append(is_correct_string)
        outputs.append([test_input[0], test_input[1], linear_combination, output])

    # review output
    num_wrong = len([is_correct for output in is_correct if is_correct == 'No'])
    output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Output'])
    print(output_frame)
    if not num_wrong:
        print('OR Perceptron Weighted well\n')


##################
# NOT PERCEPTRON #
##################

def not_perceptron(weight1, weight2, bias):
    # inputs and outputs
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    correct_outputs = [True, False, True, False]
    outputs = []
    is_correct= []

    # generate and ouput
    for test_input, correct_output in zip(test_inputs, correct_outputs):

        # linear algebra (Wx + b = y) - this is the perceptron's logic
        linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias

        # everything below is nice formatting
        output = int(linear_combination >= 0)
        # checking if the output of this perceptron is acheiving the output needed to follow the logic
        is_correct_string = 'Yes' if output == correct_output else 'No'
        is_correct.append(is_correct_string)
        outputs.append([test_input[0], test_input[1], linear_combination, output])

    # review output
    num_wrong = len([is_correct for output in is_correct if is_correct == 'No'])
    output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Output'])
    print(output_frame)
    if not num_wrong:
        print('NOT Perceptron is Weighted well')