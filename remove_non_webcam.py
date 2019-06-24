import os
from shutil import copyfile

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

d_path='/home/tiina/dummy_data/flower_photos4'
subs=get_immediate_subdirectories(d_path)
for path in [d_path+'/'+d_name for d_name in subs]:
    files=os.listdir(path)
    print(path)
    files_set=[ff for ff in files if os.path.isfile(path+'/'+ff)]
    for f in files_set:
        if f not in files_set[350:550]:
            os.remove(path+'/'+ f)

d_path='/home/tiina/vegetables/dataset2'
for path in [d_path + '/' + d_name for d_name in subs]:
    files = os.listdir(path)
    print(path)
    print(len(files))

d_path='/home/tiina/vegetables/withpaper'
subs=get_immediate_subdirectories(d_path)
for sub in subs:
    path= d_path + '/' + sub
    print(path)
    if os.path.isdir(path+'/'+sub+'Papir'):
        pathh= path+'/'+sub+'Papir'
        files=os.listdir(pathh)
        print('????????????????????????????')
        print(pathh)
        print('????????????????????????????')
        print('????????????????????????????')
        for f in [ff for ff in files if os.path.isfile(pathh+'/'+ff)]:
            print(f)
            print('HERE')
            print(type(f))
            print('Webcam' in f )
            print(os.path.isfile(f))
            if ('Webcam' in f or 'Tower' in f) and os.path.isfile(pathh+'/'+f):
                print('yes')
                copyfile(pathh+'/'+ f,path +'/'+ f)

