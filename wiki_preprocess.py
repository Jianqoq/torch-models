import sys
import unittest

import ijson
import language_tool_python
from ijson.common import ObjectBuilder
import re
import time
from langdetect import detect
from language_tool_python import LanguageTool
from layers import printProcess
import nltk
import multiprocessing as mp
import json
from layers import Preprocess
from blingfire import *


def replace_periods(match):
    s = match.group()
    return s.lower()


def replace_periods_back(match):
    s = match.group()
    s = re.sub(r"\"\s*([a-z])(.*?[.|?])", lambda m: f"\"{m.group(1).upper()}{m.group(2)}", s)
    s = re.sub(r"(\.)\s*(\w)", lambda m: f"{m.group(1)} {m.group(2).upper()}", s)
    return s


def strip(match):
    s = match.group()
    s = re.sub(r"\n", " ", s)
    s = re.sub(r" {2,}", " ", s)
    return s


def flatten_iter(lst):
    stack = list(lst)
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        else:
            yield item


def search(file, words):
    result = []
    try:
        with open(file, 'r', encoding='utf-8') as fp:
            texts = fp.read()
            for word in words:
                if word in texts:
                    article = ' '.join(flatten_iter(json.loads(texts)["words"]))
                    print("Found")
                    result.append((word, file, article))
    except:
        return []
    return result


def clean_data(to_write, path, pattern_list):
    match_pattern = re.compile(r"\"(.*?)\"")
    all_words = set([])
    text_list2 = []
    dictionary = {}
    count = 0
    with open(to_write, 'w') as f:
        try:
            with open(path, 'r', encoding='utf-8') as fp:
                texts = fp.read()
            for pattern, replace in pattern_list:
                texts = re.sub(pattern, replace, texts)
            text_list = texts.split(r"</doc>")
            period = to_raw_string('.')
            further_clean = [sentence.strip() for sentence in text_list if period in sentence]
            del text_list, texts
            for article in further_clean:
                count += 1
                if article != "":
                    text = text_to_sentences(article).split("\n")
                    copy = text[:]
                    copy2 = text[:]
                    for idx, sentence in enumerate(copy):
                        words = text_to_words(sentence.lower().replace('-', ' - ')).split(" ")
                        copy[idx] = words
                        copy2[idx] = re.sub(match_pattern, replace_periods_back, copy2[idx])
                        for word in words:
                            all_words.add(word)
                    dictionary[f"{path}_article_{count}"] = copy2
                    text_list2.append(copy)
                    del copy, copy2
            f.write(json.dumps({"words": text_list2}, indent=4))
        except Exception as e:
            print(e)
            pass
    with open(f"{to_write}_sentence.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(dictionary, indent=4))

    return all_words


def to_raw_string(text):
    # 将字符串转换为字节串(bytes)，并将所有Unicode字符转换为对应的转义序列
    raw_bytes = text.encode('unicode_escape')

    # 将字节串(bytes)转换回字符串
    raw_string = raw_bytes.decode()

    return raw_string


def get_texts(text):
    cpu_num = mp.cpu_count()
    start = 0
    gap = len(text) // cpu_num
    end = gap
    texts = []
    for i in range(cpu_num):
        texts.append(text[start:end])
        start = end
        end += gap
    return texts


def get_train(path, cores=None, pattern_list=None):
    p = printProcess()
    p.add_metrics("article")
    p.add_metrics("file")
    cpu_cores = mp.cpu_count() if cores is None else cores
    all_words = set([])
    task = []
    articles = 0
    begin = time.time()
    # char_list = [f"{chr(i)}{chr(j)}" for i in range(65, 91) for j in range(65, 91)]
    char_list = ["AA", "AB"]
    # convert a list of sentence to a list of word token list
    pattern_list = [
        (re.compile(r"(\n)[\w\s\/-]*(\n)+"), "\n"),  # remove title
        (re.compile(r"<doc.*?>\n</doc>\n"), ""),  # remove useless article
        (re.compile(r"<doc.*?>\n"), ""),  # remove useless article
        (re.compile(r"(\n)[\w\s\\]*(\n)"), "\nParagraph\n"),  # remove title
        (re.compile(r"&lt;.*?&gt;(\n)"), ""),  # remove &lt;...&gt;
        (re.compile(r"&lt;.*?&gt;"), " "),  # remove &lt;...&gt;
        (re.compile(r"\([^\w\"\'\)\]]+"), r"( "),  # change ( ; ...) to (...)
        (re.compile(r"\(\s*\)"), " "),  # remove (  ) only have space inside
        (re.compile(r"\xa0"), " "),  # change \\n\\n+ to \\n\\n
        (re.compile(r"\[\["), r" "),  # extract content from [[content]]
        (re.compile(r"\]\]"), r" "),  # extract content from [[content]]
        (re.compile(r"\["), r" "),  # extract content from [content]
        (re.compile(r"\]"), r" "),  # extract content from [content]
        (re.compile(r"\(\'\'\)"), " "),  # remove ('')
        (re.compile(r"(\n\n)+"), "\n\n"),  # change multiple \\n to one \\n
        (re.compile(r"(\\u[0-9a-fA-F]{4}|\\x[a-fA-F0-9]{2})"), r" \1 "),  # change "unicode" to " unicode "
        (re.compile(r"\u2013"), "-"),
        (re.compile(r"\"(.*?)\""), replace_periods),
        (re.compile(r"\( +"), r"("),  # change multiple space to one (  123)
        (re.compile(r" {2,}"), " "),  # change multiple space to one
        (re.compile(r"[A-Z][^!?]*[.!?]\s"), strip),
    ] if pattern_list is None else pattern_list
    with mp.Pool(processes=cpu_cores) as pool:
        pool_map = pool.starmap
        for chars in char_list:
            for i in range(100):
                if i == 0:
                    os.makedirs(rf"C:\Users\123\PycharmProjects\words2\{chars}", exist_ok=True)
                if len(task) == cpu_cores:
                    to_write = rf"C:\Users\123\PycharmProjects\words2\{chars}\wiki_{i:02}.json"
                    file_path = fr"{path}\{chars}\wiki_{i:02}"
                    task.append((to_write, file_path, pattern_list))
                    results = pool_map(clean_data, task)
                    for k in results:
                        for word in k:
                            all_words.add(word)
                    articles += len(task)
                    p.print_result(articles, file_path, begin=begin, timing=True)
                    task.clear()
                else:
                    to_write = rf"C:\Users\123\PycharmProjects\words2\{chars}\wiki_{i:02}.json"
                    file_path = fr"{path}\{chars}\wiki_{i:02}"
                    task.append((to_write, file_path, pattern_list))

    word_id, id_word, corpus = Preprocess.get_word_id(all_words)
    word_id["[CLS]"] = 1
    id_word[1] = "[CLS]"
    word_id["[PAD]"] = 0
    id_word[0] = "[PAD]"
    word_id["[MASK]"] = 2
    id_word[3] = "[MASK]"

    with open("saved_word_id.json", "w") as fp:
        json.dump(word_id, fp, indent=4)
    with open("saved_id_word.json", "w", encoding="utf-8") as fp:
        json.dump(id_word, fp, indent=4)
    with open("saved_corpus2.json", "w", encoding="utf-8") as fp:
        fp.write(json.dumps({"corpus": corpus}, indent=4))

    return word_id, id_word


def get_single_train(file_path):
    element = file_path.split("\\")
    all_words = set([])
    text_list2 = []
    chars = element[-2]
    i = element[-1]
    art = []
    # convert a list of sentence to a list of word token list
    os.makedirs(rf"single_words\{chars}", exist_ok=True)
    with open(rf"single_words\{chars}\wiki_{i:02}.json", 'w') as f:
        with open(file_path, 'r', encoding='utf-8') as fp:
            # Iterate over the dump file line by line
            original_article = fp.read()
            texts = re.sub(r"&lt;.*?&gt;", " ", original_article)
            texts = re.sub(r"\(\s*\)", " ", texts)
            texts = re.sub(r"\xa0", " ", to_raw_string(texts))
            texts = re.sub(r"\n\n+", "\n\n", texts)
            texts = re.sub(r".*?\('\).*?", " ", texts)  # (')
            texts = re.sub(r"(\(|\[)\W+\s*", r" \g<1> ", texts)  # , ; or ( ; ...)
            texts = re.sub(r"(\\u\d+)", r" \1 ", to_raw_string(texts))  # unicode
        text = texts.lower()
        text_list = texts.split("</doc>")
        for article in text_list:
            texts = re.sub(r'<doc.*?>', ' ', article)
            # texts = re.sub(r'(\n+\w*\n+)', '', texts)
            if texts != "":
                texts = text_to_sentences(texts)
                art.append(texts)
                texts = texts.split("\n")
                for idx, sentence in enumerate(texts):
                    words = text_to_words(sentence).split(" ")
                    texts[idx] = words
                    for word in words:
                        if word not in original_article:
                            raise RuntimeError
                        all_words.add(word)
                text_list2.append(texts)
        f.write(json.dumps({"words": text_list2}))
    word_id, id_word, corpus = Preprocess.get_word_id(all_words)
    word_id["[CLS]"] = 1
    id_word[1] = "[CLS]"
    word_id["[PAD]"] = 0
    id_word[0] = "[PAD]"
    word_id["[MASK]"] = 3
    id_word[3] = "[MASK]"

    with open("test_saved_word_id.json", "w", encoding="utf-8") as fp:
        fp.write(json.dumps(word_id, indent=4))
    with open("test_re_result.txt", "w", encoding="utf-8") as fp:
        fp.write(text)
    with open("test_text_to_sentence.txt", "w", encoding="utf-8") as fp:
        fp.write(str(art))
    with open("test_saved_corpus2.json", "w", encoding="utf-8") as fp:
        fp.write(json.dumps({"corpus": corpus}, indent=4))


def search_word(path, words, char_list=None, cores=None, save_result=False):
    for idx, word in enumerate(words):
        words[idx] = f"\"{word}\""
    p = printProcess()
    p.add_metrics("article")
    p.add_metrics("file")
    task = []
    cpu_cores = mp.cpu_count() if cores is None else cores
    articles = 0
    begin = time.time()
    o = []
    char_list = [f"{chr(i)}{chr(j)}" for i in range(65, 91) for j in range(65, 91)] if char_list is None else char_list
    with mp.Pool(processes=cpu_cores) as pool:
        pool_map = pool.starmap
        for chars in char_list:
            for i in range(100):
                file = fr"{path}\{chars}\wiki_{i:02}.json"
                if len(task) == cpu_cores - 2:
                    result = pool_map(search, task)
                    for k in result:
                        o += k
                    if len(o) == len(words):
                        break
                    task.clear()
                    p.print_result(articles, file, begin=begin, timing=True)
                else:
                    task.append((file, words))
                articles += 1
    if save_result:
        with open("search_result.json", "w", encoding="utf-8") as fp:
            dictionary = {}
            for word, file, article in o:
                dictionary[file] = (word, article)
            fp.write(json.dumps(dictionary, indent=4))
    return True if len(dictionary.keys()) > 0 else False


def generate_tensor_from_dict(dictionary):
    for key in dictionary:
        yield dictionary[key]


def parse_json(file):
    key = '-'
    builder = None
    for prefix, event, value in ijson.parse(file):
        if prefix == '' and event == 'map_key':  # found new object at the root
            key = value  # mark the key value
            builder = ObjectBuilder()
        elif prefix.startswith(key):  # while at this key, build the object
            builder.event(event, value)
            yield key, builder.value


if __name__ == "__main__":
    patternlist = [
        (re.compile(r"(\n)[\w\s\/-]*(\n)+"), "\n"),  # remove title
        (re.compile(r"<doc.*?>\n</doc>\n"), ""),  # remove useless article
        (re.compile(r"<doc.*?>\n"), ""),  # remove useless article
        (re.compile(r"(\n)[\w\s\\]*(\n)"), "\nParagraph\n"),  # remove title
        (re.compile(r"&lt;.*?&gt;(\n)"), ""),  # remove &lt;...&gt;
        (re.compile(r"&lt;.*?&gt;"), " "),  # remove &lt;...&gt;
        (re.compile(r"\([^\w\"\'\)\]]+"), r"( "),  # change ( ; ...) to (...)
        (re.compile(r"\(\s*\)"), " "),  # remove (  ) only have space inside
        (re.compile(r"\xa0"), " "),  # change \\n\\n+ to \\n\\n
        (re.compile(r"\[\["), r" "),  # extract content from [[content]]
        (re.compile(r"\]\]"), r" "),  # extract content from [[content]]
        (re.compile(r"\["), r" "),  # extract content from [content]
        (re.compile(r"\]"), r" "),  # extract content from [content]
        (re.compile(r"\(\'\'\)"), " "),  # remove ('')
        (re.compile(r"(\n\n)+"), "\n\n"),  # change multiple \\n to one \\n
        # (re.compile(r"\(.*?(u[0-9a-fA-F]{4}|x[a-fA-F0-9]{2}).*?\)"), " "),    # remove (... unicode ...)
        (re.compile(r"(\\u[0-9a-fA-F]{4}|\\x[a-fA-F0-9]{2})"), r" \1 "),  # change "unicode" to " unicode "
        (re.compile(r"\u2013"), "-"),
        (re.compile(r"\"(.*?)\""), replace_periods),
        (re.compile(r"\( +"), r"("),  # change multiple space to one (  123)
        (re.compile(r" {2,}"), " "),  # change multiple space to one
    ]
    #
    # texts = r"According to mother tongue percentage statistics by the Andorran Government released in 2018 :\\nMother tongue &lt;templatestyles src=\"Legend/styles.css\" /&gt;   Spanish \\n (43.2%)&lt;templatestyles src=\"Legend/styles.css\" /&gt;   Catalan \\n (35.7%)&lt;templatestyles src=\"Legend/styles.css\" /&gt;  Portuguese \\n (17.1%)&lt;templatestyles src=\"Legend/styles.css\" /&gt;   French\n (8.9%)&lt;templatestyles src=\"Legend/styles.css\" /&gt;  other (5%)\nThe historic and official language is Catalan, a Romance language. The Andorran government encourages the use of Catalan. It funds a Commission for Catalan Toponymy in Andorra (Catalan: ), and provides free Catalan classes to assist immigrants. Andorran television and radio stations use Catalan."
    # for pattern, replace in patternlist:
    #     texts = re.sub(pattern, replace, texts)
    # print(texts)
    # _word_id, _id_word = get_train(r"C:\Users\123\PycharmProjects\results", 16)
    # text = "Renewed interest in antiquity during ] the Renaissance and in private judgment during the Reformation restored elements of anti - authoritarian secularism, particularly in France."

    # matches = tool.check(text)
    # for mistake in matches:
    #     print(f"Mistake found: {mistake}, Suggested correction: ")
