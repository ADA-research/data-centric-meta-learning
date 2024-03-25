exp_name="my-experiment"

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
exp_path="$parent_path/experiments/$exp_name"
errors_path="$exp_path/logs/errors"
output_path="$exp_path/logs/outs"

architecture="resnet-18"
train_lr=0.001
train_momentum=0.9
test_size=0.2
train_epochs=50
batch_size=16
seed=1

transfer_lr=0.001
transfer_momentum=0.9
num_target_classes=5
transfer_epochs=50
folds=5

num_datasets=29