import pandas as pd
caltech="CalTech101"
cifar="CIFAR10"
mnist="MNIST"
eps_aug = lambda e: f"L2 Noise eps:{e}"
correct = "Correct"
robust = "Robust"
de_adv = "De-adversarial"
me = "Mean"
ma = "Max"
mi = "Min"
rot = "Rotate"
trans = "Translate"


data=[]
columns = ["Dataset", "Epsilon", "Augmentation", "Metric", "SampleAgg", "BatchAgg", "val"]

aug_data=[]
aug_columns = ["Dataset", "Augmentation", "Clean Accuracy", "Augmented Accuracy"]

aug_data.append([caltech, eps_aug(1), 1, 1])
aug_data.append([caltech, eps_aug(2), 1, 1])
aug_data.append([caltech, eps_aug(5), 1, 1])
aug_data.append([caltech, "Translate", 0.96875, 0.90625])
aug_data.append([caltech, "Rotate", 0.90625, 0.875])

aug_data.append([cifar, eps_aug(1), 0.84375, 0.90625])
aug_data.append([cifar, eps_aug(2), 0.84375, 0.84375])
aug_data.append([cifar, eps_aug(5), 0.84375, 0.6875])
aug_data.append([cifar, "Translate", 0.71875, 0.75])
aug_data.append([cifar, "Rotate", 0.90626, 0.75])

aug_data.append([mnist, eps_aug(1), 1, 1])
aug_data.append([mnist, eps_aug(2), 0.96875, 0.96875])
aug_data.append([mnist, eps_aug(5), 0.96875, 0.875])
aug_data.append([mnist, "Translate", 1, 0.8125])
aug_data.append([mnist, "Rotate", 1, 1])
aug_df = pd.DataFrame(data=aug_data, columns=aug_columns)


def all_same(dset, epsilon, augmentation, val=0, metrics=None, sample_aggs=None):
    d = []
    if metrics is None:
        metrics = [correct, robust, de_adv]
    if sample_aggs is None:
        sample_aggs = [me, ma, mi]
    for metric in metrics:
        for sample_agg in sample_aggs:
            for batch_agg in [me, ma, mi]:
                d.append([dset, epsilon, augmentation, metric, sample_agg, batch_agg, val])
    return d

def specify_vals(dset, epsilon, augmentation, metric, l):
    """
    l should have exactly 9 elements
    """
    d = []
    d.append([dset, epsilon, augmentation, metric, me, me, l[0]])
    d.append([dset, epsilon, augmentation, metric, me, ma, l[1]])
    d.append([dset, epsilon, augmentation, metric, me, mi, l[2]])

    d.append([dset, epsilon, augmentation, metric, ma, me, l[3]])
    d.append([dset, epsilon, augmentation, metric, ma, me, l[4]])
    d.append([dset, epsilon, augmentation, metric, ma, me, l[5]])

    d.append([dset, epsilon, augmentation, metric, mi, me, l[6]])
    d.append([dset, epsilon, augmentation, metric, mi, me, l[7]])
    d.append([dset, epsilon, augmentation, metric, mi, me, l[8]])
    return d

def specify_all_vals(dset, epsilon, augmentation, l0, l1, l2):
    d = []
    metrics = [correct, robust, de_adv]
    ls = [l0, l1, l2]
    for i, metric in enumerate(metrics):
        d.extend(specify_vals(dset, epsilon, augmentation, metric, l=ls[i]))
    return d

d = all_same(caltech, 1, eps_aug(1))
data.extend(d)
d = all_same(cifar, 1, eps_aug(1))
data.extend(d)
d = all_same(cifar, 2, eps_aug(2))
data.extend(d)
d = all_same(cifar, 5, eps_aug(5))
data.extend(d)

## Now CALTECH
d = all_same(caltech, 2, eps_aug(2), sample_aggs=[me], val=0.04347)
data.extend(d)
d = all_same(caltech, 2, eps_aug(2), sample_aggs=[ma], val=1)
data.extend(d)
d = all_same(caltech, 2, eps_aug(2), sample_aggs=[mi])
data.extend(d)

d = all_same(caltech, 5, eps_aug(5), metrics=[correct, de_adv])
data.extend(d)
d = all_same(caltech, 5, eps_aug(5), metrics=[robust], sample_aggs=[mi])
data.extend(d)
d = [[caltech, 5, eps_aug(5), robust, me, me, 0.00582], [caltech, 5, eps_aug(5), robust, me, ma, 0.03448], [caltech, 5, eps_aug(5), robust, me, mi, 0], [caltech, 5, eps_aug(5), robust, ma, me, 0.17], [caltech, 5, eps_aug(5), robust, ma, ma, 1], [caltech, 5, eps_aug(5), robust, ma, mi, 0]]
data.extend(d)

d = specify_vals(caltech, 2, rot, correct, l=[0.3176, 0.8846, 0, 0.76, 1, 0, 0, 0, 0])
data.extend(d)

d = specify_vals(caltech, 2, rot, robust, l=[0.616, 1, 0, 0.99, 1, 0 ,0.1, 1, 0])
data.extend(d)
d = specify_vals(caltech, 2, rot, de_adv, l=[0.32, 0.88, 0, 0.83, 1, 0, 0, 0, 0])
data.extend(d)

d = specify_vals(caltech, 2, trans, correct, l=[0.4257, 0.8076, 0.0384, 1, 1, 1, 0, 0, 0])
data.extend(d)

d = specify_vals(caltech, 2, trans, robust, l=[0.6657, 1, 0.0769, 1, 1, 1, 0.27, 1, 0])
data.extend(d)

d = specify_vals(caltech, 2, trans, de_adv, l=[0.4265, 0.8076, 0, 0.98, 1, 0, 0, 0, 0])
data.extend(d)

############################# MNIST ###################################
l0 = [0.042, 0.166, 0, 0.36, 1, 0, 0, 0, 0]
l1 = [0.033, 0.166 ,0, 0.33, 1, 0, 0, 0, 0]
l2 = [0.034, 0.16, 0.0, 0.29, 1, 0, 0, 0, 0]
d = specify_all_vals(mnist, 1, eps_aug(1), l0=l0, l1=l1, l2=l2)
data.extend(d)

l0 = [0.1056, 0.13, 0, 0.98, 1, 0, 0, 0, 0]
l1 = [0.13782, 0.17391, 0, 0.96, 1, 0, 0, 0, 0]
l2 = [0.099, 0.13, 0, 0.99, 1, 0, 0, 0, 0]
d = specify_all_vals(mnist, 2, eps_aug(2), l0=l0, l1=l1, l2=l2)
data.extend(d)


l0 = [0.02, 0,11, 0, 0,4, 1, 0, 0, 0, 0]
l1 = [0.0369, 0.115, 0, 0.68, 1, 0, 0, 0, 0]
l2 = [0.017, 0.15, 0, 0.37, 1, 0, 0, 0, 0]
d = specify_all_vals(mnist, 5, eps_aug(5), l0=l0, l1=l1, l2=l2)
data.extend(d)

l0 = [0.647, 0.954, 0, 0.99, 1, 0, 0, 0, 0]
l1 = [0.67, 0.954, 0.045, 1, 1, 1, 0, 0, 0]
l2 = [0.659, 0.954, 0.09, 1, 1, 1, 0, 0, 0]
d = specify_all_vals(mnist, 2, rot, l0=l0, l1=l1, l2=l2)
data.extend(d)


l0 = [0.19, 0.93, 0, 0.35, 1, 0, 0, 0, 0]
l1 = [0.7106, 1, 0.0625, 1, 1, 1, 0.3, 1, 0]
l2 = [0.1768, 0.9375, 0, 0.34, 1, 0, 0, 0, 0]
d = specify_all_vals(mnist, 2, trans, l0=l0, l1=l1, l2=l2)
data.extend(d)

############################### CIFAR10 #####################

l0 = [0.133, 0.5, 0, 0.73, 1, 0, 0, 0, 0]
l1 = [0.2463, 0.766, 0, 0.79, 1, 0, 0, 0, 0]
l2 = [0.1343, 0.5, 0, 0.64, 1, 0, 0, 0, 0]
d = specify_all_vals(cifar, 2, rot, l0=l0, l1=l1, l2=l2)
data.extend(d)

l0 = [0.2243, 0.7826, 0, 0.54, 1, 0, 0, 0, 0]
l1 = [0.47652, 1, 0, 0.7, 1, 0, 0.02, 1, 0]
l2 = [0.2017, 0.7826, 0, 0.5, 1, 0, 0, 0, 0]
d = specify_all_vals(cifar, 2, trans, l0=l0, l1=l1, l2=l2)
data.extend(d)


