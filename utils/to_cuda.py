
def to_cuda(feature_lists, use_cuda=True):
    if use_cuda:
        res = [f.cuda() for f in feature_lists]
    else:
        res = [f for f in feature_lists]

    return res
