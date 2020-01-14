from pathlib import Path

import click
import pandas as pd
# from . import functions as f
from functions import check_passed, check_columns, check_sessions, check_duplicates
current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')

@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
@click.option('--submission-file', default='submission_popular.csv', help='Submission CSV file')
@click.option('--test-file', default='test.csv', help='Test CSV file')
def main(data_path, submission_file, test_file):

    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    test_csv = data_directory.joinpath(test_file)
    subm_csv = data_directory.joinpath(submission_file)
    #
    # data_directory = 'D:\Dokumenty\Systemy_rekomendacyjne\data\\'
    # test_csv = data_directory + test_file
    # subm_csv = data_directory + submission_file

    print('Reading files...')
    df_subm = pd.read_csv(subm_csv)
    df_test = pd.read_csv(test_csv)

    print('Checking for required columns in the submission file...')
    check_cols = check_columns(df_subm)
    check_passed(check_cols)

    print('Checking for duplicate sessions in the submission file...')
    check_dupl = check_duplicates(df_subm)
    check_passed(check_dupl)

    print('Checking that all the required sessions are present in submission...')
    check_sess = check_sessions(df_subm, df_test)
    check_passed(check_sess)

    if all([check_cols, check_dupl, check_sess]):
        print('All checks passed')
    else:
        print('One or more checks failed')


if __name__ == '__main__':
    main()
