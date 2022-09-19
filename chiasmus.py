import fasttext
fasttext.FastText.eprint = lambda x: None 
import json
import time
from scipy.spatial import distance
import numpy as np
import spacy
import pickle
import re
from tqdm import tqdm
import os
ls = os.listdir
pj = os.path.join



from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor


class ChiasmusDetector: 
    def __init__(self, fasttext_model=None, verbose = False, neglist = None, conjlist = None, feature_types = None, C=1, model_type="logreg", spacy_model = None, chiasmus_regex_pattern = None, pos_blacklist=None, rating_model=None):
       
        if fasttext_model is not None:
            self.fasttext_model = fasttext.load_model('./fasttext_models/wiki.de.bin')
        else:
            self.fasttext_model = None

        if spacy_model is not None:
            self.spacy_model = spacy.load(spacy_model)

        self.neglist = neglist
        self.conjlist = conjlist
        self.feature_types = feature_types

        self.chiasmus_regex_pattern = chiasmus_regex_pattern

        self.summary = None
        self.C = C

        self.pos_blacklist = pos_blacklist
        if self.pos_blacklist is None:
            self.pos_blacklist = []

        self.model_type = model_type

        self.random_projection_matrix = np.random.rand(4, 300)

        if isinstance(rating_model, str):
            with open(rating_model, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = rating_model

        self.positive_annotations = ['a', 'fa', 'c', 'fc']

    def preprocess_text(self, text):
        assert(self.fasttext_model is not None)
        assert(self.spacy_model is not None)

        print("\tprocessing...")
        processed = self.spacy_model(text)

        print("\textracting...")
        data = []
        for p in processed:
            d = {}
            d["token"] = p.text
            d["lemma"] = p.lemma_
            d["pos"] = p.pos_
            d["dep"] = p.dep_
            #d["vectors"] = p.vector if p.has_vector else None
            d["vectors"] = self.fasttext_model[p.text]
            data.append(d)
        return data 

    def find_candidates(self, text, window_size, id_start = "", pattern = None):
        pattern = self.chiasmus_regex_pattern
        candidates = []
        if pattern == None:
            use_regex = False
        else:
            use_regex = True
        tokens = []
        lemmas = []
        pos = []
        dep = []
        vectors = []
        
        print("splitting file into lists")
        for w in tqdm(text):
            if(len(w)>0):
                tokens.append(w["token"]) 
                lemmas.append(w["lemma"])
                pos.append(w["pos"])
                dep.append(w["dep"])
                vectors.append(w["vectors"])

        for a1 in tqdm(range(len(pos)-4)):
            if use_regex and (pattern.match(pos[a1]) == None):
                continue
            if pos[a1] in self.pos_blacklist:
                continue
            for a2 in range(a1+3, min(len(pos), a1+window_size)):
                if use_regex and (pattern.match(pos[a2]) == None):
                    continue
                if pos[a2] in self.pos_blacklist:
                    continue
                for b1 in range(a1+1, a2-1):
                    if use_regex and (pattern.match(pos[b1]) == None):
                        continue
                    if pos[b1] in self.pos_blacklist:
                        continue
                    for b2 in range(b1+1, a2):
                        if use_regex and (pattern.match(pos[b2]) == None):
                            continue
                        if pos[b2] in self.pos_blacklist:
                            continue
                        if (pos[a1] == pos[a2]) and (pos[b1] == pos[b2]):
                            candidates.append({"ids": [a1, b1, b2, a2]})

        for i, c in enumerate(candidates):
            ids = c["ids"]
            c["cont_ids"] = [max(0, c["ids"][0]-5), min(len(tokens)-1, c["ids"][3]+5)]
            cont_ids = c["cont_ids"]
            c["tokens"] = tokens[cont_ids[0]:cont_ids[1]+1]
            c["lemmas"] = lemmas[cont_ids[0]:cont_ids[1]+1]
            c["pos"] = pos[cont_ids[0]:cont_ids[1]+1]
            c["dep"] = dep[cont_ids[0]:cont_ids[1]+1]
            c["vectors"] = vectors[cont_ids[0]:cont_ids[1]+1]
            c['candidate_id'] = f"{id_start}{i}"

        return candidates

    def rate_candidates(self, candidates):
        assert(self.model is not None)
        print('\tget features...')
        features = np.asarray([
                self.get_features(c) for c in tqdm(candidates)
                ])
        model = self.model
        ratings = model.decision_function(features)
        print('\trate...')
        for i, c in enumerate(tqdm(candidates)):
            c['rating'] = ratings[i]
        pass

    def run_pipeline_on_text(self, filename, text_folder="text", processed_folder="processed", candidates_folder="candidates", id_start=""):

        assert(os.path.exists(text_folder))
        assert(os.path.exists(processed_folder))
        assert(os.path.exists(candidates_folder))

        with open(pj(text_folder, filename), 'r') as f:
            text = f.read()

        # check if processed exists, if yes, don't process
        if os.path.exists(pj(processed_folder, filename)):
            with open(pj(processed_folder, filename+'.pkl'), 'rb') as f:
                processed = pickle.load(f)
        else:
            print('preprocess')
            processed = self.preprocess_text(text)
            with open(pj(processed_folder, filename)+'.pkl', 'wb') as f:
                pickle.dump(processed, f)

        # check if candidates exist, if yes, don't get them new
        if os.path.exists(pj(processed_folder, filename)):
            with open(pj(processed_folder, filename)+'.pkl', 'rb') as f:
                candidates = pickle.load(f)
        else:
            print('find candidates')
            candidates = self.find_candidates(processed, window_size=20, id_start=id_start)
            with open(pj(candidates_folder, filename)+'.pkl', 'wb') as f:
                pickle.dump(candidates, f)

        print('rate candidates')
        self.rate_candidates(candidates)
        with open(pj(candidates_folder, filename)+'.pkl', 'wb') as f:
            pickle.dump(candidates, f)
        pass

    def get_top(self, rated_candidate_file, output_file, number):
        with open(rated_candidate_file, 'rb') as f:
            candidates = pickle.load(f)
        ratings = [c['rating'] for c in candidates]
        sorting = np.argsort(ratings)[::-1]
        num_candidates = min(number, len(candidates))
        save_candidates = []
        for i in range(num_candidates):
            c = candidates[sorting[i]]
            save_candidates.append({
                'context': ' '.join(c['tokens']),
                'supporting': [
                    c['tokens'][c['ids'][0]-c['cont_ids'][0]],
                    c['tokens'][c['ids'][1]-c['cont_ids'][0]],
                    c['tokens'][c['ids'][2]-c['cont_ids'][0]],
                    c['tokens'][c['ids'][3]-c['cont_ids'][0]]],
                'supporting pos': [
                    c['pos'][c['ids'][0]-c['cont_ids'][0]],
                    c['pos'][c['ids'][1]-c['cont_ids'][0]],
                    c['pos'][c['ids'][2]-c['cont_ids'][0]],
                    c['pos'][c['ids'][3]-c['cont_ids'][0]]],
                'candidate_id': c['candidate_id'],
                'rating': c['rating']
                })

        with open(output_file, 'w') as f:
            json.dump(save_candidates, f, ensure_ascii=False, indent=4)





    def get_random_features(self, candidate):
        c = candidate 
        ids = c["ids"] 
        ia1 = ids[0]-c["cont_ids"][0] 
        ib1 = ids[1]-c["cont_ids"][0] 
        ib2 = ids[2]-c["cont_ids"][0] 
        ia2 = ids[3]-c["cont_ids"][0] 
        tokens = c["tokens"]
        lemmas = c["lemmas"]
        vectors = c["vectors"]
        pos = c["pos"]
        dep = c["dep"]


        features = []


        hardp_list = ['.', '(', ')', "[", "]"]
        softp_list = [',', ';']

        for i in [ia1, ia2, ib1, ib2]:
            for j in [ia1, ia2, ib1, ib2]:
                if j <= i:
                    continue
                v = vectors[i]-vectors[j]
                ft = self.random_projection_matrix.dot(v)
                for f in ft:
                    features.append(f)
                

        return features

    def get_dubremetz_features(self, candidate):
        c = candidate 
        ids = c["ids"] 
        ia1 = ids[0]-c["cont_ids"][0] 
        ib1 = ids[1]-c["cont_ids"][0] 
        ib2 = ids[2]-c["cont_ids"][0] 
        ia2 = ids[3]-c["cont_ids"][0] 
        tokens = c["tokens"]
        lemmas = c["lemmas"]
        vectors = c["vectors"]
        pos = c["pos"]
        dep = c["dep"]

        conjlist = self.conjlist
        neglist = self.neglist

        features = []


        hardp_list = ['.', '(', ')', "[", "]"] 
        softp_list = [',', ';']


        # Basic

        num_punct = 0
        for h in hardp_list:
            if h in tokens[ ia1+1 : ib1 ]: num_punct+=1
            if h in tokens[ ib2+1 : ia2 ]: num_punct+=1
        features.append(num_punct)

        num_punct = 0
        for h in hardp_list:
            if h in tokens[ ia1+1 : ib1 ]: num_punct+=1
            if h in tokens[ ib2+1 : ia2 ]: num_punct+=1
        features.append(num_punct)

        num_punct = 0
        for h in hardp_list:
            if h in tokens[ ib1+1 : ib2 ]: num_punct+=1
        features.append(num_punct)

        rep_a1 = -1
        if lemmas[ia1] == lemmas[ia2]:
            rep_a1 -= 1
        rep_a1 += lemmas.count(lemmas[ia1])
        features.append(rep_a1)

        rep_b1 = -1
        if lemmas[ib1] == lemmas[ib2]:
            rep_b1 -= 1
        rep_b1 += lemmas.count(lemmas[ib1])
        features.append(rep_b1)

        rep_b2 = -1
        if lemmas[ib1] == lemmas[ib2]:
            rep_b2 -= 1
        rep_b2 += lemmas.count(lemmas[ib2])
        features.append(rep_b2)

        rep_a2 = -1
        if lemmas[ia1] == lemmas[ia2]:
            rep_a2 -= 1
        rep_a2 += lemmas.count(lemmas[ia2])
        features.append(rep_b2)

        # Size

        diff_size = abs((ib1-ia1) - (ia2-ib2))
        features.append(diff_size)

        toks_in_bc = ia2-ib1
        features.append(toks_in_bc)

        # Similarity

        exact_match = ([" ".join(tokens[ia1+1 : ib1])] == [" ".join(tokens[ib2+1 : ia2])])
        features.append(exact_match)

        same_tok = 0
        for l in lemmas[ia1+1 : ib1]:
            if l in lemmas[ib2+1 : ia2]: same_tok += 1
        features.append(same_tok)

        sim_score = same_tok / (ib1-ia1)
        features.append(sim_score)

        num_bigrams = 0
        t1 = " ".join(tokens[ia1+1 : ib1])
        t2 = " ".join(tokens[ib2+1 : ia2])
        s1 = set()
        s2 = set()
        for t in range(len(t1)-1):
            bigram = t1[t:t+2]
            s1.add(bigram)
        for t in range(len(t2)-1):
            bigram = t2[t:t+2]
            s2.add(bigram)
        for b in s1:
            if b in s2: num_bigrams += 1
        bigrams_normed = (num_bigrams/max(len(s1)+1, len(s2)+1))
        features.append(bigrams_normed)

        num_trigrams = 0
        t1 = " ".join(tokens[ia1+1 : ib1])
        t2 = " ".join(tokens[ib2+1 : ia2])
        s1 = set()
        s2 = set()
        for t in range(len(t1)-2):
            trigram = t1[t:t+3]
            s1.add(trigram)
        for t in range(len(t2)-2):
            trigram = t2[t:t+3]
            s2.add(trigram)
        for t in s1:
            if t in s2: num_trigrams += 1
        trigrams_normed = (num_trigrams/max(len(s1)+1, len(s2)+1))
        features.append(trigrams_normed)

        same_cont = 0
        t1 = set(tokens[ia1+1:ib1])
        t2 = set(tokens[ib2+1:ia2])
        for t in t1:
            if t in t2: same_cont += 1
        features.append(same_cont)

        # Lexical clues

        conj = 0
        for c in conjlist:
            if c in tokens[ib1+1:ib2]+lemmas[ib1+1:ib2]:
                conj = 1
        features.append(conj)


        neg = 0
        for n in neglist:
            if n in tokens[ib1+1:ib2]+lemmas[ib1+1:ib2]:
                neg = 1
        features.append(neg)


        # Dependency score

        if dep[ib1] == dep[ia2]:
            features.append(1)  
        else: 
            features.append(0)

        if dep[ia1] == dep[ib2]:
            features.append(1)  
        else: 
            features.append(0)

        if dep[ib1] == dep[ib2]:
            features.append(1)  
        else: 
            features.append(0)

        if dep[ia1] == dep[ia2]:
            features.append(1)  
        else: 
            features.append(0)

        # Return
        return features


    def get_embedding_features(self, candidate):
        assert(self.neglist is not None)
        assert(self.conjlist is not None)
        c = candidate 
        ids = c["ids"] 
        ia1 = ids[0]-c["cont_ids"][0] 
        ib1 = ids[1]-c["cont_ids"][0] 
        ib2 = ids[2]-c["cont_ids"][0] 
        ia2 = ids[3]-c["cont_ids"][0] 
        tokens = c["tokens"]
        lemmas = c["lemmas"]
        vectors = c["vectors"]
        pos = c["pos"]
        dep = c["dep"]

        conjlist = self.conjlist
        neglist = self.neglist

        features = []


        hardp_list = ['.', '(', ')', "[", "]"]
        softp_list = [',', ';']

        for i in [ia1, ia2, ib1, ib2]:
            if vectors[i] is not None:
                assert(len(vectors[i] > 1))
            for j in [ia1, ia2, ib1, ib2]:
                if j <= i:
                    continue
                if vectors[i] is None or vectors[j] is None:
                    features.append(1)
                else:
                    features.append(distance.cosine(vectors[i], vectors[j]))
        return np.asarray(features)

    def get_lexical_features(self, candidate):
        c = candidate 
        ids = c["ids"] 
        ia1 = ids[0]-c["cont_ids"][0] 
        ib1 = ids[1]-c["cont_ids"][0] 
        ib2 = ids[2]-c["cont_ids"][0] 
        ia2 = ids[3]-c["cont_ids"][0] 
        tokens = c["tokens"]
        lemmas = c["lemmas"]
        vectors = c["vectors"]
        pos = c["pos"]
        dep = c["dep"]

        conjlist = self.conjlist
        neglist = self.neglist

        features = []


        hardp_list = ['.', '(', ')', "[", "]"]
        softp_list = [',', ';']

        for i in [ia1, ia2, ib1, ib2]:
            for j in [ia1, ia2, ib1, ib2]:
                if j <= i:
                    continue
                features.append(int(lemmas[i] == lemmas[j]))

        return np.asarray(features)

    def get_features(self, candidate):
        assert(self.feature_types is not None)
        funcs = {
                "embedding": self.get_embedding_features,
                "lexical": self.get_lexical_features,
                "dubremetz": self.get_dubremetz_features,
                "random": self.get_random_features
                }
        features = [funcs[ft](candidate) for ft in self.feature_types]
        return np.concatenate(features, axis=0)

    def _preprocess_training_data(self, data):
        assert(self.fasttext_model is not None)
        #print("compute vectors if needed")
        for d in data:
            if "vectors" in d:
                continue
            tokens = d["tokens"]
            vectors = [self.fasttext_model[t] for t in tokens]
            d["vectors"] = vectors

        #print("turn into numpy arrays")


        x = []
        y = []
        for d in data:
            x.append(self.get_features(d))
            y.append(1 if d["annotation"] in self.positive_annotations else 0)


        x = np.asarray(x)
        y = np.asarray(y)

        return x, y



    def _train(self, x, y):
        model = None
        if self.model_type == "logreg":
            model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        C = self.C))
        elif self.model_type.lower() == 'rbf svm':
            model = make_pipeline(
                    StandardScaler(),
                    SVC(
                        class_weight="balanced",
                        gamma='scale',
                        max_iter=1000,
                        C = self.C))
        elif self.model_type.lower() == 'decisiontree':
            model = make_pipeline(
                    StandardScaler(),
                    DecisionTreeRegressor()
                    )

        else:
            print("ERROR:",self.model_type, 'does not exist')
        assert(model is not None)
        model.fit(x, y)
        scores = model.decision_function(x)
        ap = average_precision_score(y, scores, average='macro')
        return model, ap

    def _load_data(self, filename):
        if ".json" == filename[-5:]:
            with open(filename, 'r') as f:
                return json.load(f)
        elif ".pickle" == filename[-7:]:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            return []

    def train(self, training_file, keep_model = True):
        data = self._load_data(training_file)
        x, y = self._preprocess_training_data(data)
        model, train_ap = self._train(x, y)
        if keep_model:
            self.model = model

    def train_with_crossval(self, training_file, num_runs=5):
        data = self._load_data(training_file)
        x, y = self._preprocess_training_data(data)
        kf = StratifiedKFold(n_splits=num_runs)

        aps_test = []
        aps_train = []

        for train_index, test_index in kf.split(x, y):
            x_train = x[train_index, :]
            y_train = y[train_index]
            x_test = x[test_index, :]
            y_test = y[test_index]

            model, ap_train = self._train(x_train, y_train)
            scores = model.decision_function(x_test)
            ap_test = average_precision_score(y_test, scores, average='macro')
            aps_train.append(ap_train)
            aps_test.append(ap_test)

        ap_train = np.mean(np.asarray(aps_train))
        ap_test = np.mean(np.asarray(aps_test))
        ap_train_std = np.std(np.asarray(aps_train))
        ap_test_std = np.std(np.asarray(aps_test))

        self.summary = {
                "ap_train": ap_train,
                "ap_test": ap_test,
                "ap_train_std": ap_train_std,
                "ap_test_std": ap_test_std
                }

    def print_summary(self):
        assert self.feature_types is not None
        print('feature types')
        for ft in self.feature_types:
            print("\t", ft)
        if self.summary is not None:
            print(f"average precisions\n\ttrain:\t{self.summary['ap_train']:.2f}+-{self.summary['ap_train_std']:.2f}\n\ttest:\t{self.summary['ap_test']:.2f}+-{self.summary['ap_test_std']:.2f}")
        else:
            print("model has not been trained")
    
    def eval(self):
        pass

    def comp_arr_binary(self, model, X, Y):
        rrs = [0, 0]
        divisor = [0, 0]
        Y_pred = model.predict(X)
        for y, y_pred in zip(Y, Y_pred):
            divisor[y] += 1
            if y == y_pred:
                rrs[y] += 1

        rrs[0] /= divisor[0]
        rrs[1] /= divisor[1]
        return np.mean(rrs)
