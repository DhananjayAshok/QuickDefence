# Experiment

- Experiment 1: Reproduce randomized smoothing graph
  - Three plots has increasing sizes of the attacks on x-axis
  - On the y-axis, we record both accuracies and confidence (p_a - p_b, p_a = probability of most likely class, p_b probability of second most likely class) when models are given pgd_l2 attacks
  - Four models (combinations)
    - Robust(trained w/ all 3 data augmentations) & non-robust (trained w/o data augmentation) models
    - With test time majority voting (also all 3 data augmentations) & without
  - Anticipated conclusion:
    - For robust and non-robust model, majority voting doesn’t change accuracies
    - Robust training works, there is some region where robust model has higher accuracy than non-robust
- Experiment 2:
  - Using models trained from experiment 1, varies test time data augmentation strength on x-axis (three augmentations: noise, rotation, affine)
  - On y-axis, record both accuracy and confidence
  - 2 models \times 2 types of input
    - Input: clean image & adversarial image
    - Model: robust vs. non-robust
  - Anticipated conclusion:
    - For robust & non-robust model, p_a-p_b for successful adversarial image is much higher than clean image. This explains why majority voting won’t do anything for robust model b/c the certified range is bigger for adversarial image
    - With increasing augmentation strength, for both models, the number of times adversarial label is predicted for adversarial image drops to random slower than accuracy with clean image. This reinforces that successful attacks with robust model becomes more robust.

## Slide Conclusions

- General Augmentations help!
- How do general augmentations help?
  - Noise augmentations drive clean predictions to random faster than adversarial predictions
  - High rotation/translation drive adversarial predictions to random, but less effect on clean image
  - Why does this make sense? Gradient-based signal is not shift-invariant.
- Does training help?
  - Not clear. Help w/ low attack norm. But worse w/ high attack norm

## Report Conclusiosn

- Do general augmentations help?

  - Noise only works for L2 attack, fails for all other attacks (model=A_attack=PGDL2)
  - All TTA works great for L2 (model=A_attack=PGDL2), weak but non-zero performance on most other attacks (model=A_attack=FFGSM), don't work for translationally-invariant attack (model=A_attack=TIFGSM)
  - It seems training with augmentations and TTA should work for non-translation invariant attack always, but we found that with a smaller model (Resnet-18) and other attacks, the test-time augmentation don't help that much.

- Was TTA or training helping?

  - TTA always help

    - TTA-alone helped compared to w/o TTA (a1, model=U_attack=TIFGSM)
    - Above we see TTA on top of training with augmentation also helped.

  - Mixed effect with augmented training
    - Augmented training alone helped a bit for PGDL2 (a2, tta=U_attack=PGDL2), but effect disappear for other attack (TIFGSM)
    - All-TTA: Augmented training trades off initial accuracy with adversarial robustness. We suspect attacks also become somewhat augmentation-invariant when they are generated on models trained on augmentations (tta=A_attack=PGDL2)

- Can augmentations drown out attacks?
  - Noise fail to drown out all the attacks we tried
  - Rotation/Translation are successful w/ PGD L2 when the model is untrained\
    - Makes Intuitive sense that Rotation/Translation are better than noise
    - (Compare Noise w/ Rotation when model is untrained)
  - Interestingly, when the model is trained, rotation/translation no longers wash out attacks. We suspect this is b/c the PGD attacks become more tranlsationally invariant since they are generated to attack a translational-invariant model.
    - In fact, we see this in a2 for PGDL2. Training with augmentation trades off
    - This pattern is flipped for FFGSM. We are not sure why.
  - Finally, both rotation and translation fail to drown out attacks that are explicitly trained to be translation-invariant TIFGSM, and other attacks, PGD, TPGD
