from pathlib import Path

import click
import pandas as pd

# from . import functions as f
from functions import get_popularity, get_submission_target, explode, explode2, calc_recommendation, calc_recommendation2, add_sum_of_properties

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')


@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
def main(data_path):

    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    train_csv = data_directory.joinpath('train.csv')
    test_csv = data_directory.joinpath('test.csv')
    subm_csv = data_directory.joinpath('submission_popular22.csv')

    print(f"Reading {train_csv} ...")
    df_train = pd.read_csv(train_csv)

    # OSTATNIE DNI DANYCH
    min_time = (df_train['timestamp'].max() - 24*60*60*1)  # godziny*minuty*sekundy*dni
    mask = df_train['timestamp'] > min_time
    df_train = df_train[mask]

    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv)

    print("Get popular items...")
    df_popular = get_popularity(df_train)

    # print("Add sum of properties...")
    # df_popular_and_sum_properties = add_sum_of_properties(df_popular, df_train)

    print("Identify target rows...")
    df_target = get_submission_target(df_test)

    print("Get recommendations...")
    df_expl = explode(df_target, "impressions")
    df_out = calc_recommendation(df_expl, df_popular)

    print(f"Writing {subm_csv}...")
    df_out.to_csv(subm_csv, index=False)

    print("Finished calculating recommendations.")


if __name__ == '__main__':
    main()
