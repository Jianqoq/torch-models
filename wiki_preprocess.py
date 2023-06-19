import ijson
from ijson.common import ObjectBuilder
import re
import time
from layers import printProcess
import multiprocessing as mp
import json
from layers import Preprocess
from blingfire import *


def to_lower(match):
    s = match.group()
    s = re.sub(r"\s+(\"\s+)\w", lambda m: f"{m.group(1).replace(' ', '')}", s)
    s = re.sub(r"\w(\s+\")\s+", lambda m: f"{m.group(1).replace(' ', '')}", s)
    return s.lower()


def replace_periods_parentheses_back(match, origin, lower):
    s = match.group()
    if '[period]' in s:
        index = lower.index(s)
        s = origin[index]
    return s


def replace_periods_back(text: str, symbol):
    replaced = re.sub(fr"([.?!])\s+{symbol}[^.?!\w]*([a-z])(.*?[.?!])",
                      lambda m: f"{m.group(1)} \"{m.group(2).upper()}{m.group(3)}",
                      text)
    text = re.sub(r"([.!?])\s*(\w)", lambda m: f"{m.group(1)} {m.group(2).upper()}", replaced)
    return text


def remove_nested(s, symbol1, symbol2):
    match = re.finditer(r'\[\[File:', s)
    result = [i.start() for i in match]
    match_strings = []
    if result:
        for m in result:
            stack = []
            current = s[m:]
            i = 0
            FLAG = False
            while i < len(current):
                if current[i:i+len(symbol1)] == symbol1:
                    stack.append(symbol1)
                    i += 2
                elif current[i:i+len(symbol2)] == symbol2:
                    i += 2
                    stack.pop()
                    if not stack:
                        FLAG = True
                        break
                else:
                    i += 1
            if FLAG:
                match_strings.append(s[m:m + i])
    if len(match_strings) == 1:
        s = s.replace(match_strings[0], "")
    else:
        for i in match_strings:
            s = s.replace(i, "")
    return s


def find_nested(s, to_find, symbol1, symbol2):
    match = re.finditer(to_find, s)
    result = [i.start() for i in match]
    match_strings = []
    if result and symbol1 != symbol2:
        for m in result:
            stack = []
            current = s[m:]
            i = 0
            FLAG = False
            while i < len(current):
                if current[i:i+len(symbol1)] == symbol1:
                    stack.append(symbol1)
                    i += len(symbol1)
                elif current[i:i+len(symbol2)] == symbol2:
                    i += len(symbol2)
                    stack.pop()
                    if not stack:
                        FLAG = True
                        break
                else:
                    i += 1
            if FLAG:
                match_strings.append(s[m:m + i])
    elif result and symbol1 == symbol2:
        for m in result:
            stack = []
            current = s[m:]
            i = 0
            FLAG = False
            while i < len(current):
                if current[i:i+len(symbol1)] == symbol1:
                    stack.append(symbol1)
                    i += len(symbol1)
                elif current[i:i+len(symbol2)] == symbol2:
                    i += len(symbol2)
                    stack.pop()
                    if not stack:
                        FLAG = True
                        break
                else:
                    i += 1
            if FLAG:
                match_strings.append(s[m:m + i])
    return match_strings


def strip(match):
    s = match.group()
    s = re.sub(r"\n", " ", s)
    s = re.sub(r" {2,}", " ", s)
    return s


def strip_quote(match):
    s = match.group()
    s = re.sub(r"\" +| +\"", "\"", s)
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
    all_words = set([])
    text_list2 = []
    dictionary = {}
    count = 0
    with open(to_write, 'w') as f:
        try:
            with open(path, 'r', encoding='utf-8') as fp:
                texts = fp.read()
                texts = remove_nested(texts, "[[", "]]")
            for pattern, replace in pattern_list:
                texts = re.sub(pattern, replace, texts)
            origin = find_nested(texts, r'\(', '(', ')')
            lower = []
            cache = None
            for i in origin:
                if '.' in i:
                    cache = i.lower().replace('.', '[period]')
                    texts = texts.replace(i, cache)
                if cache is not None:
                    lower.append(cache)
                else:
                    lower.append(i)
                cache = None
            text_list = texts.split(r"</doc>")
            further_clean = [article.strip() for article in text_list if '.' in article]
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
                        if '[period]' in copy2[idx]:
                            match = find_nested(copy2[idx], r'\(', '(', ')')
                            for i in match:
                                if i in lower:
                                    index = lower.index(i)
                                    copy2[idx] = copy2[idx].replace(i, origin[index])
                        if '\"' in copy2[idx]:
                            copy2[idx] = replace_periods_back(copy2[idx], r'\"')
                        if '\'' in copy2[idx]:
                            copy2[idx] = replace_periods_back(copy2[idx], r'\'')
                        copy2[idx] = copy2[idx].replace(' i ', ' I ').replace('i. E.', 'i.e.')
                        for word in words:
                            all_words.add(word)

                    dictionary[f"{path}_article_{count}"] = copy2
                    text_list2.append(copy)
                    del copy, copy2
            f.write(json.dumps({"words": text_list2}, indent=4))
        except Exception as e:
            raise Exception(e)
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


def strip_space(match):
    sub = re.sub(r'\s+', '', match.group(1))
    return f"{sub} "


def get_train(path, cores=None, pattern_list=None):
    p = printProcess()
    p.add_metrics("article")
    p.add_metrics("file")
    cpu_cores = mp.cpu_count() if cores is None else cores
    all_words = set([])
    task = []
    articles = 0
    begin = time.time()
    char_list = [f"{chr(i)}{chr(j)}" for i in range(65, 91) for j in range(65, 91)]
    # char_list = ["AA"]
    # convert a list of sentence to a list of word token list
    pattern_list = [
        (re.compile(r"(\n)[\w\s/-]*(\n)+"), "\n"),  # remove title
        (re.compile(r"<doc.*?>\n</doc>\n"), ""),  # remove useless article
        (re.compile(r"<doc.*?>\n"), ""),  # remove useless article
        (re.compile(r"&lt;.*?&gt;(\n)"), ""),  # remove &lt;...&gt;
        (re.compile(r"&lt;.*?&gt;"), " "),  # remove &lt;...&gt;
        (re.compile(r"[^\w)\]\"\'(\[]+([;,.?!])"), r"\g<1>"),  # hello:, ; to hello;
        (re.compile(r"\([^\w\"\')\]\[]+"), r"("),  # change ( ; ...) to (...)
        (re.compile(r"\{[^\w\"\')\]\[]+"), r"{"),  # change ( ; ...) to (...)
        (re.compile(r"\(\s*\)"), " "),  # remove (  ) only have space inside
        (re.compile(r"\xa0"), " "),  # change \\n\\n+ to \\n\\n
        (re.compile(r"\"\[\["), "\""),  # extract content from [[content]]
        (re.compile(r"]]\""), "\""),  # extract content from [[content]]
        (re.compile(r"\[\["), " "),  # extract content from [[content]]
        (re.compile(r"]]"), " "),  # extract content from [[content]]
        (re.compile(r"\"\["), "\""),  # extract content from [content]
        (re.compile(r"]\""), "\""),  # extract content from [content]
        (re.compile(r"\["), " "),  # extract content from [content]
        (re.compile(r"]"), " "),  # extract content from [content]
        (re.compile(r"\(\'\'\)"), " "),  # remove ('')
        (re.compile(r"\(\"\"\)"), " "),  # remove ("")
        (re.compile(r"\(\'\)"), " "),  # remove (')
        (re.compile(r"(\n\n)+"), "\n\n"),  # change multiple \\n to one \\n
        (re.compile(r"(\\u[0-9a-fA-F]{4}|\\x[a-fA-F0-9]{2})"), r" \1 "),  # change "unicode" to " unicode "
        (re.compile(r"\u2013"), "-"),  # from \u2013 to -
        (re.compile(r"\"(.*?)\""), to_lower),  # make everything inside "" lower
        (re.compile(r"\s+(\'[^.?!\w]*([a-zA-Z]){2,}.*?[!?.]\')"), to_lower),  # make everything inside '' lower
        (re.compile(r"\( +"), r"("),  # change multiple space to one (  123)
        (re.compile(r"\{ +"), r"{"),  # change multiple space to one (  123)
        (re.compile(r"([^\s\w\"\'.?!%)\]]\s*)+([])}])"), r"\g<2>"),  # (: ) -> ()
        (re.compile(r"\s+?([?:,.])\s+?"), r"\1 "),  # strip space around ? : , .
        (re.compile(r"\s+([]|)])"), r"\1"),  # (hello ) -> (hello)
        (re.compile(r" {2,}"), " "),  # change multiple space to one
        (re.compile(r"[A-Z][^!?]*[.!?]\s"), strip),
    ] if pattern_list is None else pattern_list
    with mp.Pool(processes=cpu_cores) as pool:
        pool_map = pool.starmap
        try:
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
        except:
            pass

    word_id, id_word, corpus = Preprocess.get_word_id(all_words)
    word_id["[CLS]"] = 1
    id_word[1] = "[CLS]"
    word_id["[PAD]"] = 0
    id_word[0] = "[PAD]"
    word_id["[MASK]"] = 2
    id_word[2] = "[MASK]"
    word_id["[SEP]"] = 3
    id_word[3] = "[SEP]"

    with open("saved_word_id.json", "w") as fp:
        json.dump(word_id, fp, indent=4)
    with open("saved_id_word.json", "w", encoding="utf-8") as fp:
        json.dump(id_word, fp, indent=4)
    with open("saved_corpus2.json", "w", encoding="utf-8") as fp:
        fp.write(json.dumps({"corpus": corpus}, indent=4))

    return word_id, id_word


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
    pattern_list = [
        (re.compile(r"(\n)[\w\s/-]*(\n)+"), "\n"),  # remove title
        (re.compile(r"<doc.*?>\n</doc>\n"), ""),  # remove useless article
        (re.compile(r"<doc.*?>\n"), ""),  # remove useless article
        (re.compile(r"&lt;.*?&gt;(\n)"), ""),  # remove &lt;...&gt;
        (re.compile(r"&lt;.*?&gt;"), " "),  # remove &lt;...&gt;
        (re.compile(r"[^\w)\]\"\'(\[]+([;,.?!])"), r"\g<1>"),  # hello:, ; to hello;
        (re.compile(r"\([^\w\"\')\]\[]+"), r"("),  # change ( ; ...) to (...)
        (re.compile(r"\{[^\w\"\')\]\[]+"), r"{"),  # change ( ; ...) to (...)
        (re.compile(r"\(\s*\)"), " "),  # remove (  ) only have space inside
        (re.compile(r"\xa0"), " "),  # change \\n\\n+ to \\n\\n
        (re.compile(r"\"\[\["), "\""),  # extract content from [[content]]
        (re.compile(r"]]\""), "\""),  # extract content from [[content]]
        (re.compile(r"\[\["), " "),  # extract content from [[content]]
        (re.compile(r"]]"), " "),  # extract content from [[content]]
        (re.compile(r"\"\["), "\""),  # extract content from [content]
        (re.compile(r"]\""), "\""),  # extract content from [content]
        (re.compile(r"\["), " "),  # extract content from [content]
        (re.compile(r"]"), " "),  # extract content from [content]
        (re.compile(r"\(\'\'\)"), " "),  # remove ('')
        (re.compile(r"\(\"\"\)"), " "),  # remove ("")
        (re.compile(r"\(\'\)"), " "),  # remove (')
        (re.compile(r"(\n\n)+"), "\n\n"),  # change multiple \\n to one \\n
        (re.compile(r"(\\u[0-9a-fA-F]{4}|\\x[a-fA-F0-9]{2})"), r" \1 "),  # change "unicode" to " unicode "
        (re.compile(r"\u2013"), "-"),  # from \u2013 to -
        (re.compile(r"\"(.*?)\""), to_lower),  # make everything inside "" lower
        (re.compile(r"\s+(\'[^.?!\w]*([a-zA-Z]){2,}.*?[!?.]\')"), to_lower),  # make everything inside '' lower
        (re.compile(r"\( +"), r"("),  # change multiple space to one (  123)
        (re.compile(r"\{ +"), r"{"),  # change multiple space to one (  123)
        (re.compile(r"([^\s\w\"\'.?!%)\]]\s*)+([])}])"), r"\g<2>"),  # (: ) -> ()
        (re.compile(r"\s+?([?:,.])\s+?"), r"\1 "),  # strip space around ? : , .
        (re.compile(r"\s+([]|)}])"), r"\1"),  # (hello ) -> (hello)
        (re.compile(r" {2,}"), " "),  # change multiple space to one
        (re.compile(r"[A-Z][^!?]*[.!?]\s"), strip),
    ]
    _word_id, _id_word = get_train(r"C:\Users\123\PycharmProjects\results", 16)
    # t = 'The name "Apollo"—unlike the related older name "Paean"—is generally not found in the Linear B (Mycenean Greek) texts, although there is a possible attestation in the lacunose form "]pe-rjo-[" (Linear B: ]-[) on the KN E 842 tablet, though it has also been suggested that the name might actually read "Hyperion" ([u]-pe-rjo-[ne]).'
    # for i, replace in pattern_list:
    #     t = re.sub(i, replace, t)
    # print(t)