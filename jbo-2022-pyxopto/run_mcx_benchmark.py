import subprocess

def run(file_json, n_photons):

    args = [
        'mcx',
        '-f', '{}'.format(file_json),  # input file
        '-e', '0.0001',                # trigger Russian roulette
        '-d', '0',                     # do not save photon info
        '-D', 'P',                     # print progress bar
        '-O', 'E',                     # energy deposit
        '-n', '{}'.format(n_photons),  # number of photons
        '-A', '1'                      # auto threads and threads per block
    ]

    subprocess.run(args)

run('mcxyz_benchmark_input.json', n_photons=2e9)