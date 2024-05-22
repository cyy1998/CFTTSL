import os

def main():
    dir_lis=os.listdir('E:\\3d_retrieval\\Dataset\\Shrec_13\\12_views\\13_view_render_img')
    with open('classname.txt','w') as f:
        for i in range(len(dir_lis)):
            f.write(str(i)+':'+dir_lis[i]+'\n')

if __name__ == '__main__':
    main()