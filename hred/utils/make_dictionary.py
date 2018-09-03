from os import listdir
from os.path import isfile, join
import re
import numpy as np
import operator

personal_path ="/Users/leehayeon/peanut/"

path1 = personal_path+"hred/data/conversation/"
path2 = personal_path+"hred/data/drama_data/"
path3= personal_path+"hred/data/corpus_data/processed_data_without_colon/"
path4 = personal_path+"hred/data/etc/"
path = [path1, path2, path3, path4]

file_lists1= [f for f in listdir(path1) if isfile(join(path1, f))]
file_lists2= [f for f in listdir(path2) if isfile(join(path2, f))]
file_lists3= [f for f in listdir(path3) if isfile(join(path3, f))]
file_lists4= [f for f in listdir(path4) if isfile(join(path4, f))]
file_lists=[file_lists1, file_lists2, file_lists3, file_lists4]


pattern = "[^ ㄱ - ㅎ|가-힣 ]+"
regex = re.compile(pattern)

dot_pattern = "[^ ㄱ - ㅎ|가-힣_. ]+"
dot_regex = re.compile(dot_pattern)


#make dictionary.txt file
#and separate test and training file by multiple 10000 lines
def make_dictionary(txt_name):
        dictionary={}
        npy_arr =[]

        dict_expt = ['_P_', '_S_', '_E_', '_U_', ' ' ]
        for token in dict_expt:
                dictionary[token] = len(dictionary)

        txt_file = open(personal_path+"hred/data/preprocessed_all.txt", "r")
        test_file = open(personal_path+'hred/data/preprocessed_all_test.txt', 'w')
        training_file = open(personal_path+'hred/data/preprocessed_all_training.txt', 'w')

        is_test =True
        idx = 0

        for line in txt_file.readlines():
                idx+=1

                if line=='\n':
                        if is_test : 
                                test_file.write(line)
                        else:
                                training_file.write(line)
                        continue

                line = line.strip()

                '''
                for token in line.split(' '):
                        if token not in dictionary:
                                dictionary[token] = len(dictionary)
                '''

                for token in line.split(' '):
                        if token == '\n':
                                continue

                        if token not in dictionary:
                                dictionary[token] = len(dictionary)

                
                npy_line =[]
                for token in line.split(' '):
                        if token == '\n':
                                continue
                        npy_line.append(dictionary[token])
                npy_arr.append(npy_line)
                
                
                if is_test : 
                        test_file.write(line+'\n')
                else:
                        training_file.write(line+'\n')
                
                if is_test and len(npy_arr) % 100==0 and len(npy_arr)>=30000:
                        np.save(personal_path +"hred/data/dict_idx_"+txt_name.split('_')[1]+"_test.npy", npy_arr)
                        is_test = False
                        npy_arr = []
                if not is_test and len(npy_arr) % 100 ==0 and len(npy_arr)>=70000:
                        break

        txt_file.close()
        test_file.close()
        training_file.close()
       
        np.save(personal_path +"hred/data/dict_idx_"+txt_name.split('_')[1]+"_training.npy", npy_arr)


        # making dictionary
        dict_file = open(personal_path +"hred/data/dictionary.txt","w")
        for token in dictionary:
                dict_file.write(token+'\n')
        dict_file.close()





def make_idxnpy(txt_name, npy_name):
        
        dictionary={}

        dict_file = open(personal_path+'hred/data/dictionary.txt', "r")
        dict_expt = ['_P_', '_S_', '_E_', '_U_', ' ' ]
        for idx, token in enumerate(dict_expt):
                dictionary[token] = idx

        for idx, token in enumerate(dict_file.readlines()):
                dictionary[token.strip()] = idx
        dict_file.close()

        
        txt_file =  open(personal_path+"hred/data/"+txt_name, "r")
        npy_arr = []
        for line in txt_file.readlines():
                if line == '\n':
                        continue

                npy_line =[]

                for token in line.strip().split(' '):

                        npy_line.append(dictionary[token])
                npy_arr.append(npy_line)

        txt_file.close()

        np.save(personal_path+'hred/data/dict_idx_'+npy_name+'.npy', npy_arr)



def preprocessing_data(path, file):
        global regex
        all_txt = ""


        max_len = 100
        limit_len = 35


        if file==".DS_Store" or ('form' not in file):
                return "", 0
                
        
        dialogue=""
        read_f = open(path+file, "r")
        ss_in_dialogue =0 
        dlg_end =False
        line_num=0
        for line in read_f.readlines():
                #removing special characters
                if line =='\n' or dlg_end:
                        if ss_in_dialogue > 1:
                                all_txt += dialogue+'\n\n'
                                line_num += ss_in_dialogue
                        dlg_end = False
                        ss_in_dialogue=0
                        dialogue=""
                        continue

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
                        new_line = regex.sub(" ", line).strip()
                        while '  ' in new_line:
                                new_line= new_line.replace('  ', ' ')
                        new_line += '\n'

                if len(new_line.split(' '))> limit_len:
                        dlg_end= True
                        continue

                dialogue+= new_line
                ss_in_dialogue+=1


        all_txt += dialogue+'\n\n'
        read_f.close()
        
        return all_txt, line_num


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



def format_file(path, file_list):
        global dot_regex
        global path2

        for i in range(len(file_list)):
                if file_list[i]==".DS_Store" or "form" in file_list[i]:
                        continue

                print('formating\t', file_list[i])

                write_format = open(path +'form_'+file_list[i], "w")
                read_f = open(path+file_list[i], "r")

                for origin_line in read_f.readlines():
                        origin_line = origin_line.replace('!','.', len(origin_line)).replace('?','.', len(origin_line))
                        origin_line = remove_expl(origin_line)
                        ##regular expression_allowing period
                        reg_line = dot_regex.sub(" ", origin_line)

                        bef_line = reg_line

                        while '  ' in reg_line:
                                reg_line= reg_line.replace('  ', ' ')

                        while '..' in reg_line:
                                reg_line= reg_line.replace('..', '.')

                        if '  ' in reg_line:
                                print('before', bef_line)
                                print('after', reg_line)
                        
                        if reg_line == '.':
                                continue



                        write_format.write(reg_line+'\n')
                write_format.close()
                read_f.close()


# matching format of files
def formating_files():
        for i, path_i in enumerate(path):
                format_file(path_i, file_lists[i])

def check_freq():
        dictionary = {}
        for i in range(len(path)):
                for j in range(len(file_lists[i])):
                        if file_lists[i][j] == ".DS_Store" or ('form' not in file_lists[i][j]):
                                continue
                        txt_file = open(path[i]+file_lists[i][j], "r")
                        for line in txt_file.readlines():
                                for token in line.strip().split(' '):
                                        if token not in dictionary :
                                                dictionary[token] = 0
                                        dictionary[token]+=1

        sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1))
        print('dictionary', sorted_dict)

def check_theCHAR(theChar, path):

        dictionary ={}
        txt_file = open(path, "r")
        num =0 
        for line in txt_file.readlines():
                line = line.strip()
                num += line.count(theChar)
                if line.count(theChar) > 0 :
                        print(line)
                for token in line.split(' '):
                        if theChar in token:
                                if token not in dictionary:
                                        dictionary[token] =0
                                dictionary[token] += 1

        sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1))
        print(sorted_dict)

        return num

#write into 1 file and removing too long sentences
def gather_and_preprocess(txt_name):
        total_path =[]
        for i in range(len(path)):
                for j in range(len(file_lists[i])):
                        if file_lists[i][j]==".DS_Store" or ('form' not in file_lists[i][j]):
                                continue

                        if i==0:
                                continue

                        total_path.append([path[i], file_lists[i][j]])

        total_path = np.array(total_path)
        np.random.shuffle(total_path)

        line_num =0
        gather=""

        idx =0 

        while idx < len(total_path):
                pair = total_path [idx]
                txt, num = preprocessing_data(pair[0], pair[1])
                line_num += num
                gather += txt
                idx+=1

        
        write_f = open(personal_path+"hred/data/"+txt_name, "w")
        write_f.write(gather)
        write_f.close()
        


def main():
        formating_files()
        gather_and_preprocess("preprocessed_all.txt")
        check_freq()
        make_dictionary("preprocessed_all")

        make_idxnpy("preprocessed_all_test.txt", "all_test")
        make_idxnpy("preprocessed_all_training.txt", "all_training")


if __name__ == "__main__":
        main()



