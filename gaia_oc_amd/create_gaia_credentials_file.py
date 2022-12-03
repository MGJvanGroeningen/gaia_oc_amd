import os
import argparse
from getpass import getpass


def create_gaia_credentials_file(save_path='./gaia_credentials'):
    print('Create a credentials file for ESA Gaia archive')
    username = input('Username : ')
    password = getpass('Password : ')

    with open(save_path, 'w') as f:
        f.write(username + "\n")
        f.write(password)

    # no read or write for 'others'
    os.chmod(save_path, 0o660)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', nargs='?', type=str, default='./data',
                        help='Directory where credentials file will be saved.')
    args_dict = vars(parser.parse_args())

    create_gaia_credentials_file(args_dict['save_dir'])
