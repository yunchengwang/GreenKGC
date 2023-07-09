import os
import random
import numpy as np


def load_dataset(data_dir, reciprocal=False):
    entity2id = dict()
    relation2id = dict()

    with open(os.path.join(data_dir, "entities.dict"), 'r+', encoding="utf-8") as f:
        for line in iter(f):
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(data_dir, "relations.dict"), 'r+', encoding="utf-8") as f:
        for line in iter(f):
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    if typing:
        all_ent_types = []
        with open(os.path.join(data_dir, "types.txt"), 'r+', encoding="utf-8") as f:
            for line in iter(f):
                ent_type = line.strip()
                all_ent_types.append(entity2id[ent_type])

    n_entity = len(entity2id)
    n_relation = len(relation2id)

    data = {"train": [], "valid": [], "test": []}

    with open(os.path.join(data_dir, "train.txt"), 'r+', encoding="utf-8") as f:
        for line in iter(f):
            head, relation, tail = line.strip().split('\t')
            head, tail = entity2id[head], entity2id[tail]
            relation = relation2id[relation]
            if typing and relation != 0:
                continue
            data['train'].append((head, relation, tail))
            if reciprocal:
                data['train'].append((tail, relation + n_relation, head))

    data['train'] = list(set(data['train']))

    with open(os.path.join(data_dir, "valid.txt"), 'r+', encoding="utf-8") as f:
        for line in iter(f):
            head, relation, tail = line.strip().split('\t')
            head, tail = entity2id[head], entity2id[tail]
            relation = relation2id[relation]
            data['valid'].append((head, relation, tail))
            if reciprocal:
                data['valid'].append((tail, relation + n_relation, head))

    data['valid'] = list(set(data['valid']))

    with open(os.path.join(data_dir, "test.txt"), 'r+', encoding="utf-8") as f:
        for line in iter(f):
            head, relation, tail = line.strip().split('\t')
            head, tail = entity2id[head], entity2id[tail]
            relation = relation2id[relation]
            data['test'].append((head, relation, tail))
            if reciprocal:
                data['test'].append((tail, relation + n_relation, head))

    data['test'] = list(set(data['test']))

    data['complete'] = list(set(data['train'] + data['valid'] + data['test']))

    return n_entity, n_relation, data


def calc_lcw_index(train_triples_by_relation, k=5):
    error = dict()
    n_triples = dict()
    train_triples = []
    for r in train_triples_by_relation:
        error[r] = 0
        n_triples[r] = len(train_triples_by_relation[r])
        for h, t in train_triples_by_relation[r]:
            train_triples.append((h, r, t))

    random.shuffle(train_triples)

    for i in range(k):
        valid, train = train_triples[i * len(train_triples) // k:(i + 1) * len(train_triples) // k], \
                       train_triples[: i * len(train_triples) // k] + train_triples[(i + 1) * len(train_triples) // k:]
        relation_range = dict()
        for h, r, t in train:
            if r not in relation_range:
                relation_range[r] = set()
            relation_range[r].add(t)

        for h, r, t in valid:
            if r in set(relation_range):
                if t not in relation_range[r]:
                    error[r] += 1

    lcw_index = dict()

    for r in train_triples_by_relation:
        lcw_index[r] = 1.0 - error[r] / n_triples[r]

    return lcw_index


def sigmoid(x):
    return 1/(1 + np.exp(-x))
