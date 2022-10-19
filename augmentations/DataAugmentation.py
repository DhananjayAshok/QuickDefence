import numpy.random as random


class DataAugmentation:
    """
    Parent class for all types of augmentations
    """
    def __init__(self, sequence):
        """

        :param sequence: When sequence(input=inp) is called returns processed augmented input
        """
        self.sequence = sequence

    def __call__(self, inp):
        """

        :param inp: input to the neural network we seek to protect
        :return: augmented input x
        """
        inp = self.sequence(inp)
        return inp


class RandomSequenceSlice(DataAugmentation):
    """
    Given a list of different data augmentations, will perform them at random. 
    """
    def __init__(self, operations, randomization=None):
        """

        :param operations: list of sequence data augmentations to perform
        :param randomization: either None or a list of probabilities where len(randomization) = len(operations)
                              operation i will take place with probability randomization[i]
                              if len(randomization) < len(operations) we use randomization[0] for extra elements
        """
        self.operations = operations
        self.randomization = randomization

    def __call__(self, inp):
        """
        Sequentially performs data augmentations

        :param inp:
        :return: augmented input x
        """
        randomization = self.randomization
        # assert randomization is None or (randomization != [] and all([0<=r<=1 for r in randomization]))
        x = inp
        for i, operation in self.operations:
            if randomization is not None:
                p = randomization[0]
                if i < len(randomization):
                    p = randomization[i]
                if p > random.random():
                    x = self.operations[i](x)
            else:
                x = self.operations[i](x)
        return x