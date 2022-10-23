import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('username', nargs='?', type=str,
                        help='ESA Gaia Archive username')
    parser.add_argument('password', nargs='?', type=str,
                        help='ESA Gaia Archive password')
    parser.add_argument('--save_dir', nargs='?', type=str, default='.',
                        help='Directory where credentials file will be saved.')
    args_dict = vars(parser.parse_args())

    username = args_dict['username']
    password = args_dict['password']
    save_dir = args_dict['save_dir']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    credentials_file = os.path.join(save_dir, 'gaia_credentials')

    with open(credentials_file, 'w') as f:
        f.write(username + "\n")
        f.write(password)

    # no read or write for 'others'
    os.chmod(credentials_file, 0o660)
