# QuickDefence

Main problems:

1. I have been using foolbox to get ImageNet images and adversarial attacks, but it seems like there is a mismatch between the pretrained pytorch Imagenet models and this data, the pytorch models get very low accuracy on foolbox image net and so we are unable to benchmark the adversarial robustness of the model or determine the defence efficacy. 

To do:

1. Fix a dataset, get multiple pretrained models for that dataset, benchmark their accuracy on the dataset
2. Fix a few adversarial attacks (whether Foolbox or directly) and implement them, benchmark the adversarial robustness of above models on the dataset we got above (not foolbox Imagenet)
3. Run the defence tool and benchmark succes rates. Run the adversarial density functions to judge proof of concept. Come up with reports
