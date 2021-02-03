def count_n_parameters(model):
    """
    count number of learnable parameters of a model
    :param model: subclass of nn.Module.
    :return: int. number of parameters
    """
    pp = 0
    for p in list(model.parameters()):
        n = 1
        for s in list(p.size()):
            n = n*s
        pp += n
    return pp

