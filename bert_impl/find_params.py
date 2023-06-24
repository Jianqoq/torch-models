import torch
from sklearn.model_selection import RandomizedSearchCV
from tokenizers.implementations import BertWordPieceTokenizer
from bert_impl.Bert import Bert, generate_list_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    tokenizer = BertWordPieceTokenizer("custom/vocab.txt")
    char_list = [f"{chr(i)}{chr(j)}" for i in range(65, 91) for j in range(65, 91)]
    char_list = char_list[:char_list.index('FT') + 1]
    files = [fr"C:\Users\123\PycharmProjects\words2\{chars}\wiki_{i:02}.json_sentence.json_json.json.json" for i in
             range(100) for chars in
             char_list]
    train = files[:9]
    test = [files[-1]]
    files2 = [fr"C:\Users\123\PycharmProjects\words2\{chars}\wiki_{i:02}.json_sentence.json" for i in range(100) for
              chars in
              char_list]
    total = len(files)
    count = 0
    _embedding_dim = 768
    _hidden_size = 3072
    _num_head = 12
    _out_dim = 512
    max_epoch = 10
    batch = 130
    _num_layers = 12
    # _words = generate_data(files)
    train_data = generate_list_data(train)
    bert = Bert(tokenizer.get_vocab_size(), _embedding_dim, _hidden_size, _num_head, 128, _num_layers, tokenizer)
    param_dist = {
        'corpus': [tokenizer.get_vocab_size()],
        'embedding_dim': [384, 516, 786, 1152],
        'hidden_size': [1920, 2560, 3072],
        'num_head': [4, 12],
        'max_length': [128],
        'num_layers': [12],
        'n_segments': [2],
        'learning_rate': [1e-4]
    }
    fit_params = {
        'X_test': generate_list_data(test),
        'y_test': torch.nn.CrossEntropyLoss(ignore_index=0),
        'batch_size': batch,
        'max_epoch': max_epoch,
        'tokenizer': tokenizer
    }
    random_search = RandomizedSearchCV(
        bert, param_distributions=param_dist, n_iter=10, cv=9, n_jobs=1)
    random_search.fit(train_data, **fit_params)
    best_params = random_search.best_params_
    print("Best parameters found: ", best_params)
    results = random_search.cv_results_
    print(results)