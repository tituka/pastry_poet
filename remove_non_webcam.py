import os
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

d_path='/home/tiina/vegetables/dataset3'
subs=get_immediate_subdirectories(d_path)
for path in [d_path+'/'+d_name for d_name in subs]:
    files=os.listdir(path)
    print(path)
    for f in [ff for ff in files if os.path.isfile(path+'/'+ff)]:
        print(f)
        if not ('Webcam' in f or 'Tower' in f):
            os.remove(path+'/'+ f)

d_path='/home/tiina/vegetables/dataset2'
for path in [d_path + '/' + d_name for d_name in subs]:
    files = os.listdir(path)
    print(path)
    print(len(files))
