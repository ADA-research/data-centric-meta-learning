import openml
import argparse


def main(dataset_id: int):
    dataset = openml.datasets.get_dataset(
        dataset_id, download_data=True, download_all_files=True)
    print(dataset.data_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', required=True)

    args = parser.parse_args()
    main(args.dataset_id)
