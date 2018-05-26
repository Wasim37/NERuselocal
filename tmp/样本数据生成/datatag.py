#-*-encoding=utf8-*-
import csv
import jieba
import jieba.posseg as pseg
import re,os

fout1=open('example.dev','w',encoding='utf8')
fout2=open('example.test','w',encoding='utf8')
fout3=open('example.train','w',encoding='utf8')

flag=0
fuhao= ['；','。','?','？','!','！',';']
biaoji=['DIS','SYM','SGN','TES','DRU','SUR','PRE','PT','Dur','TP','REG','ORG','AT','PSB','DEG','FW','CL']

dics=csv.reader(open("DICT_NOW.csv",'r',encoding='utf8'))
for row in dics:
    if flag==0:
        flag=1
        continue
    if len(row)==2:
        jieba.add_word(row[0].strip(),tag=row[1].strip())
        #jieba.suggest_freq(segment)

files=os.listdir("F:\\git_pro\\ml_project\\NERuselocal_github\\样本数据生成\\original_data\\")
path_dir="F:\\git_pro\\ml_project\\NERuselocal_github\\样本数据生成\\original_data\\"
#split_num1=sum([len(open(path_dir+file,'r',encoding='utf8').readlines()) for file in files if "txtoriginal" in file])
#if split_num%15==0:
    #index=str(1)
#elif split_num%15>0 and split_num%15<4:
    #index=str(2)
#else:
    #index=str(3)
split_num=0
for file in files:
    # 只处理文件名包含 txtoriginal 的文件
    if "txtoriginal" in file:
        fp=open(path_dir+file,'r',encoding='utf8')
        lines=[line for line in fp]
        for line in lines:
            split_num+=1
            words=pseg.cut(line) # jieba分词
            for key,value in words: 
                    print(key) # 词语
                    print(value) # 词性
                    if value.strip() and key.strip():
                        if value not in biaoji:
                            value='O'
                            for achar in key.strip():
                                # 按行数来划分数据集，比例为 1:2:13
                                if split_num%15<2:
                                    index=str(1)
                                elif split_num%15>1 and split_num%15<4:
                                    index=str(2)
                                else:
                                    index=str(3)                                   
                                if achar and achar.strip() in fuhao:
                                    string=achar+" "+value.strip()+"\n"+"\n"
                                 
                                    if index=='1':                               
                                        fout1.write(string)
                                    elif index=='2':
                                        fout2.write(string)
                                    elif index=='3':
                                        fout3.write(string)
                                    else:
                                        pass
                                elif achar.strip() and achar.strip() not in fuhao:
                                    string = achar + " " + value.strip() + "\n"
                                    if index=='1':
                                        fout1.write(string)
                                    elif index=='2':
                                        fout2.write(string)
                                    elif index=='3':
                                        fout3.write(string)
                                    else:
                                        pass
                                else:
                                    continue
                                
                        elif value.strip()  in biaoji:
                            begin=0
                            for char in key.strip():
                                if begin==0:
                                    begin+=1
                                    string1=char+' '+'B-'+value.strip()+'\n'
                                    if index=='1':
                                        fout1.write(string1)
                                    elif index=='2':
                                        fout2.write(string1)
                                    elif index=='3':
                                        fout3.write(string1)
                                    else:
                                        pass
                                else:
                                    string1 = char + ' ' + 'I-' + value.strip() + '\n'
                                    if index=='1':
                                        fout1.write(string1)
                                    elif index=='2':
                                        fout2.write(string1)
                                    elif index=='3':
                                        fout3.write(string1)
                                    else:
                                        pass
                        else:
                            continue                        
fout1.close()
fout2.close()
fout3.close()
#if __name__=='__main__':
    #for k,v in pseg.cut("今天中午吃什么"):
        #print(k)
        #print(v)