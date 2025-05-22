from scripts.pre_train import pre_train
from scripts.fine_tune import fine_tune
from scripts.features import calculate_features

# make sure the BCT_Mini and PLK_Mini dataset is available locally (see data folder for instructions)

pre_train(dataset='BCT_Mini', exp_path='example_experiment', epochs=2)

fine_tune(foundation='BCT_Mini', target='PLK_Mini',
          exp_path='example_experiment', epochs=2, folds=2)

calculate_features(dataset='PLK_Mini', exp_path='example_experiment')
calculate_features(dataset='BCT_Mini', exp_path='example_experiment')
