class Attack:

    def __call__(self, model, input_batch, intended_labels=None, true_labels=None):
        raise NotImplementedError
