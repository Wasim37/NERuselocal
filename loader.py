import os
import re
import codecs

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num+=1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        # print(list(line))
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word= line.split( )
            assert len(word) == 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    
    序列标记常见标记方案：
    http://nathanlvzs.github.io/Several-Tagging-Schemes-for-Sequential-Tagging.html
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    #print("Found %i unique words (%i in total)" % (
    #    len(dico), sum(len(x) for x in chars)
    #))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    
    f=open('tag_to_id.txt','w',encoding='utf8')
    f1=open('id_to_tag.txt','w',encoding='utf8')
    tags=[]
    for s in sentences:
        ts=[]
        for char in s:
            tag=char[-1]
            ts.append(tag)
        tags.append(ts)
    
    #tags1 = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    #print("Found %i unique named entity tags" % len(dico))
    for k,v in tag_to_id.items():
        f.write(k+":"+str(v)+"\n")
    for k,v in id_to_tag.items():
        f1.write(str(k) + ":" + str(v) + "\n")
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        #print(sentences)
        string = [w[0] for w in s]
        # 字在字典中的位置
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        # 字的长度特征
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        #string 训练数据集的第一个句子
        #['无', '长', '期', '0', '0', '0', '年', '0', '0', '月', '0', '0', '日', '出', '院', '记', '录', '患', '者', '姓', '名', '：', '闫', 'X', 'X', '性', '别', '：', '男', '年', '龄', '：', '0', '0', '岁', '入', '院', '日', '期', '：', '0', '0', '0', '0', '年', '0', '0', '月', '0', '0', '日', '0', '0', '时', '0', '0', '分', '出', '院', '日', '期', '：', '0', '0', '0', '0', '年', '0', '0', '月', '0', '0', '日', '0', '0', '时', '0', '0', '分', '共', '住', '院', '0', '0', '天', '。']
        #chars 句子中每个字符在字典中的位置
        #[6, 297, 109, 2, 2, 2, 34, 2, 2, 54, 2, 2, 50, 29, 11, 138, 204, 40, 37, 205, 182, 4, 1309, 78, 78, 10, 168, 4, 359, 34, 236, 4, 2, 2, 176, 48, 11, 50, 109, 4, 2, 2, 2, 2, 34, 2, 2, 54, 2, 2, 50, 2, 2, 43, 2, 2, 30, 29, 11, 50, 109, 4, 2, 2, 2, 2, 34, 2, 2, 54, 2, 2, 50, 2, 2, 43, 2, 2, 30, 338, 71, 11, 2, 2, 122, 5]
        #segs 句子每个词语的长度 0 表示单个字 1表示词语的开头 2表示词语的中间词 3表示词语的结尾
        #[0, 1, 3, 1, 2, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 1, 3, 1, 3, 0, 0, 1, 3, 1, 3, 0, 0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 0, 1, 2, 2, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 0, 1, 2, 2, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 1, 3, 1, 3, 0, 0]
        #tags 句子中对应的tag标记在字典（tag_to_id）中的位置
        #[0, 43, 44, 0, 0, 0, 0, 0, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 16, 0, 0, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0]
        data.append([string, chars, segs, tags])

    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    #print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)

