import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('username', nargs='?', type=str,
                        help='ESA Gaia Archive username')
    parser.add_argument('password', nargs='?', type=str,
                        help='ESA Gaia Archive password')
    parser.add_argument('--data_dir', nargs='?', type=str, default='data',
                        help='Directory where data (e.g. cone searches, source sets) '
                             'and results will be saved and retrieved.')

    args_dict = vars(parser.parse_args())
    data_dir = args_dict['data_dir']
    if not os.path.exists(data_dir):
        default_data_dir = os.path.join(os.getcwd(), 'data')
        print(f'Using default save path {default_data_dir}')
        if not os.path.exists(default_data_dir):
            os.mkdir(default_data_dir)
        data_dir = default_data_dir

    username = args_dict['username']
    password = args_dict['password']

    credentials_file = os.path.join(data_dir, 'gaia_credentials')

    with open(credentials_file, 'w') as f:
        f.write(username + "\n")
        f.write(password)

    # no read or write for 'others'
    os.chmod(credentials_file, 0o660)
