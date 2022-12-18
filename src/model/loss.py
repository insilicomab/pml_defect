from pytorch_metric_learning import losses, distances, regularizers


def get_arcfaceloss():
    distance = distances.CosineSimilarity()
    regularizer = regularizers.RegularFaceRegularizer()
    loss = losses.ArcFaceLoss(
        num_classes=2,
        embedding_size=512,
        margin=28.6,
        scale=64,
        weight_regularizer=regularizer, 
        distance=distance
    )
    sampler = None
    mining_funcs = dict()

    return loss, sampler, mining_funcs