#for file I/O funcs
def data_writer(file_name="fit_data.dat", contents="contents hoge piyo", **kwargs):
    options = {"path":"./"}
    options.update(kwargs)
    path_w = options["path"]+file_name
    with open(path_w, mode='w') as f:
        f.write(contents)
    print("File Name:", file_name)

def fit_file_reader(file_name, **kwargs):
    options = {"path":"./"}
    options.update(kwargs)
    path_r = options["path"]+file_name

    try:
        with open(path_r) as f:
            line = f.readlines()
    
        tau = (line[1].strip()).split(":")[1]
        xc = (line[2].strip()).split(":")[1]
        yc = (line[3].strip()).split(":")[1]
        r = (line[4].strip()).split(":")[1]
        fr = (line[5].strip()).split(":")[1]
        Qr = (line[6].strip()).split(":")[1]
        Qc = (line[7].strip()).split(":")[1]
        phi_0 = (line[8].strip()).split(":")[1]
    except FileNotFoundError :
        print("Cannot found fit parameters file.")
    return tau, xc, yc, r, fr, Qr, Qc, phi_0