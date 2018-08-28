from os import listdir
from os.path import isfile, join
import re
import numpy as np

personal_path ="/Users/leehayeon/peanut/"


path1 = personal_path+"hred/data/corpus_data/processed_data_without_colon/"
path2 = personal_path+"hred/data/conversation/"
path3 = personal_path+"hred/data/drama_data/"
path4 = personal_path+"hred/data/etc/"
path = [path1, path2, path3, path4]

file_lists1= [f for f in listdir(path1) if isfile(join(path1, f))]
file_lists2= [f for f in listdir(path2) if isfile(join(path2, f))]
file_lists3= [f for f in listdir(path3) if isfile(join(path3, f))]
file_lists4= [f for f in listdir(path4) if isfile(join(path4, f))]
file_lists=[file_lists1, file_lists2, file_lists3, file_lists4]

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

        for line in txt_file.readlines():
                if line=='\n':
                        continue

                line = line.strip()

                for token in line.split(' '):
                        if token not in dict:
                                dict[token] = len(dict)


                npy_line =[]
                for token in line.split(' '):
                        npy_line.append(dict[token])
                npy_arr.append(npy_line)
        
        dict_expt = ['_S_', '_E_', '_U_', '_P_']
        for token in dict_expt:
                dict[token] = len(dict)

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
                
                print('gathering\t', file_list[i])
                
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
                                new_line = regex.sub(" ", line).strip()+'\n'

                        if len(new_line.split(' '))> limit_len:
                                dlg_end= True
                                continue

                        dialogue+= new_line
                        ss_in_dialogue+=1

                all_txt += dialogue+'\n\n'
                read_f.close()
        
        return all_txt

def remove_expl(str):
        while '(' in str:
                start_idx = str.index('(')
                if ')' not in str:
                                print('err : ' +str)
                                return ''
                end_idx = str.index(')')
                new = ''
                new+=str[:start_idx]
                if end_idx+1 <= len(str):
                        new += str[end_idx+1:]
                str = new               

        return str.strip()



def format_file(path, file_list, write_f=None):
        global dot_regex
        global path2

        for i in range(len(file_list)):
                if file_list[i]==".DS_Store" or "form" in file_list[i]:
                        continue

                #print('formating\t', file_list[i])

                dialogue=""
                write_format = open(path +'form_'+file_list[i], "w")
                read_f = open(path+file_list[i], "r")

                for origin_line in read_f.readlines():
                        origin_line = origin_line.replace('!','.', len(origin_line)).replace('?','.', len(origin_line))
                        origin_line = remove_expl(origin_line)
                        ##regular expression_allowing period
                        reg_line = dot_regex.sub(" ", origin_line)

                        while '  ' in reg_line:
                                reg_line= reg_line.replace('  ', ' ')
                        while '..' in reg_line:
                                reg_line= reg_line.replace('..', '.')

                        if reg_line == '.':
                                continue



                        write_format.write(reg_line+'\n')
                write_format.close()
                read_f.close()


# matching format of files
def formating_files():
        for i, path_i in enumerate(path):
                format_file(path_i, file_lists[i])

#write into 1 file and removing too long sentences
def gather_and_preprocess():
        write_f = open(personal_path+"hred/data/preprocessed_all.txt", "w")
        gather=""
        for i, path_i in enumerate(path):
                gather+=preprocessing_data(path_i, file_lists[i])
        write_f.write(gather)
        write_f.close()


formating_files()
gather_and_preprocess()
make_dictionary()


