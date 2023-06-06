import json
import sys
import time
import unittest

from language_tool_python import LanguageTool

from layers import printProcess


class MyTest(unittest.TestCase):

    def test_errors(self):
        with LanguageTool('en-US') as tool:
            with open(r"C:\Users\123\PycharmProjects\words2\AA\wiki_00.json_sentence.json", "r", encoding="utf-8") as fp:
                rule_list = ["MORFOLOGIK_RULE_EN_US"]
                p = printProcess()
                p.add_metrics("article")
                p.add_metrics("Checking sentence")
                dic = json.load(fp)
                begin = time.time()
                error_sentence = {}
                for key in dic:
                    sentence_list = dic[key]
                    for sentence in sentence_list:
                        matches = tool.check(sentence)
                        p.print_result(key, sentence, begin=begin, timing=True)
                        if matches:
                            if len(matches) == 1 and matches[0].ruleId not in rule_list:
                                error_sentence[len(error_sentence)] = (
                                    key,
                                    sentence,
                                    tool.correct(sentence),
                                    {
                                        f"match_{idx}": [
                                            i.ruleId,
                                            i.category,
                                            i.message,
                                            {
                                                f"replacement_{index}": k for index, k in enumerate(i.replacements)
                                            }
                                        ]
                                        for idx, i in enumerate(matches)
                                    }
                                )
                            else:
                                store = False
                                for i in matches:
                                    if i.ruleId not in rule_list:
                                        store = True
                                if store:
                                    error_sentence[len(error_sentence)] = (
                                        key,
                                        sentence,
                                        tool.correct(sentence),
                                        {
                                            f"match_{idx}": [
                                                i.ruleId,
                                                i.message,
                                                {
                                                    f"replacement_{index}": k for index, k in enumerate(i.replacements)
                                                }
                                            ]
                                            for idx, i in enumerate(matches) if i.ruleId not in rule_list
                                        }
                                    )
                with open(r"error_container.json", "r", encoding="utf-8") as f:
                    try:
                        past = json.load(f)
                    except json.decoder.JSONDecodeError:
                        past = []
                    if past and len(past) < len(error_sentence):
                        tool.close()
                        self.assertGreater(len(past), len(error_sentence))
                    elif past:
                        with open(r"error_container.json", "w", encoding="utf-8") as file_pointer:
                            json.dump(error_sentence, file_pointer, indent=4)
                            tool.close()
                            self.assertGreater(len(past), len(error_sentence))
                    else:
                        with open(r"error_container.json", "w", encoding="utf-8") as file_pointer:
                            json.dump(error_sentence, file_pointer, indent=4)
                            self.assertGreater(1, 0)
