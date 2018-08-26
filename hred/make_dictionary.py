from os import listdir
from os.path import isfile, join
import re
import numpy as np

personal_path ="/Users/leehayeon/"

path1 = personal_path+"hred/data/corpus_data/processed_data_without_colon/"
path2 = personal_path+"hred/data/conversation/"
path3 = personal_path+"hred/data/drama_data/"

file_lists1= [f for f in listdir(path1) if isfile(join(path1, f))]
file_lists2= [f for f in listdir(path2) if isfile(join(path2, f))]
file_lists3= [f for f in listdir(path3) if isfile(join(path3, f))]

dict = {}

pattern = "[^ ㄱ - ㅎ|가-힣 ]+"
regex = re.compile(pattern)

dot_pattern = "[^ ㄱ - ㅎ|가-힣_. ]+"
dot_regex = re.compile(dot_pattern)

def make_dictionary():
        txt_file =  open(personal_path+"hred/data/preprocessed_all.txt", "r")
        dict_file = open(personal_path +"hred/data/dictionary.txt","w")
        dict={}

        npy_arr =[]
        print('in make_dictionary')
        for line in txt_file.readlines():
                if line=='\n':
                        continue

                for token in line.split(' '):
                        if token not in dict:
                                dict[token] = len(dict)

                npy_line =[]
                for token in line.split(' '):
                        npy_line.append(dict[token])
                npy_arr.append(npy_line)
                break

        for token in dict:
                dict_file.write(token+'\n')

        dict_file.close()
        txt_file.close()

        np.save(personal_path +"hred/data/dict_idx.npy", npy_arr)

def preprocessing_data(path, file_list):
        global regex
        all_txt = ""

        max_len = 100
        limit_len = 35
        for i in range(len(file_list)):
                if file_list[i]==".DS_Store" or ('form' not in file_list[i]):
                        continue
                
                print(file_list[i])
                
                dialogue=""
                read_f = open(path+file_list[i], "r")
                ss_in_dialogue =0 
                dlg_end =False
                for line in read_f.readlines():
                        
                        #removing special characters
                        if line =='\n' or dlg_end:
                                if ss_in_dialogue > 1:
                                        all_txt += dialogue+'\n\n'
                                dlg_end = False
                                ss_in_dialogue=0
                                dialogue=""
                                continue

                        #dlg_end = False:
                        line_len = len(line.split(' '))
                        if line_len > max_len:
                                dlg_end= True
                                continue

                                sentences = [s for s in line.strip().split('.') if s !='' and s!=' ' ]

                                first_ss = sentences[0]
                                last_ss = sentences[-1]
                                new_line = first_ss+' '+last_ss+'\n'
                                
                                while '  ' in new_line:
                                        new_line= new_line.replace('  ', ' ')

                        else:
                                new_line = regex.sub("", line).strip()+'\n'

                        if len(new_line.split(' '))> limit_len:
                                dlg_end= True
                                continue

                        dialogue+= new_line
                        ss_in_dialogue+=1

                read_f.close()
        
        return all_txt


def format_files(path, file_list, write_f=None):
        global dot_regex
        global path2

        for i in range(len(file_list)):
                if file_list[i]==".DS_Store" or "form" in file_list[i]:
                        continue

                print(file_list[i])

                dialogue=""
                write_format = open(path +'form_'+file_list[i], "w")
                read_f = open(path+file_list[i], "r")

                for origin_line in read_f.readlines():
                        origin_line = origin_line.replace('!','.', len(origin_line)).replace('?','.', len(origin_line))
                        
                        ##regular expression_allowing period
                        reg_line = dot_regex.sub("", origin_line)

                        while '  ' in reg_line:
                                reg_line= reg_line.replace('  ', ' ')
                        while '..' in reg_line:
                                reg_line= reg_line.replace('..', '.')
                        if reg_line == '.':
                                continue



                        write_format.write(reg_line+'\n')
                write_format.close()
                read_f.close()

read_1= read_2= read_3 =""

format_files(path1, file_lists1)
format_files(path2, file_lists2)
format_files(path3, file_lists3)

write_f = open(personal_path+"hred/data/preprocessed_all.txt", "w")

write_1= preprocessing_data(path1, file_lists1)
write_2= preprocessing_data(path2, file_lists2)
write_3= preprocessing_data(path3, file_lists3)

write_f.write(write_1+'\n\n')
write_f.write(write_2+'\n\n')
write_f.write(write_3+'\n\n')

write_f.close()

make_dictionary()



