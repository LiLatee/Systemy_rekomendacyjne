from pathlib import Path

import click

# from . import functions as f
from functions import score_submissions, get_reciprocal_ranks

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')

@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
@click.option('--submission-file', default='xgb_gic_lic_wosh_lf350_lr002_v2_all_ut_1mln_all.csv', help='Submission CSV file')
@click.option('--ground-truth-file', default='ground_truth.csv', help='Ground truth CSV file')
def main(data_path, submission_file, ground_truth_file):

    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    gt_csv = data_directory.joinpath(ground_truth_file)
    subm_csv = data_directory.joinpath(submission_file)

    mrr = score_submissions(subm_csv, gt_csv, get_reciprocal_ranks)

    print(f'Mean reciprocal rank: {mrr}')


if __name__ == '__main__':
    main()
