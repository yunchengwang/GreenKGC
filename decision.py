import gc
import os
import logging
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import xgboost as xgb
from utils import load_dataset, calc_lcw_index


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Code for training and testing Module 3 in GreenKGC',
        usage='decision.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--dataset', default='FB15k-237', type=str)
    parser.add_argument('--emb_dir', default='pretrained', type=str)
    parser.add_argument('-o', '--output_dir', default='output', type=str)
    parser.add_argument('--pretrained_emb', default='RotatE_pruned', type=str)
    parser.add_argument('-d', '--dim', default=100, type=int)
    parser.add_argument('--max_depth', default=5, type=int)
    parser.add_argument('--negative_size', default=64, type=int)
    parser.add_argument('--n_estimators', default=2000, type=int)
    parser.add_argument('--neg_sampling', default='naive')
    parser.add_argument('--num_group', default=3, type=int)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--lcwa', action='store_true')
    parser.add_argument('--lcw_threshold', default=0.0, type=float)

    return parser.parse_args(args)


def set_logger(args):
    log_path = "{}/{}_{}_{}".format(args.output_dir, args.pretrained_emb, args.dataset, args.dim)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, '{}.log'.format(args.neg_sampling))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info(xgb.__version__)
    logging.info('Augment: {}'.format(args.augment))
    logging.info('Negative sampling: {}'.format(args.neg_sampling))
    logging.info('# of negative samples: {}'.format(args.negative_size))
    logging.info('max depth: {}'.format(args.max_depth))
    logging.info('# of estimators: {}'.format(args.n_estimators))
    logging.info('LCWA-based link prediction: {}'.format(args.lcwa))
    if args.lcwa:
        logging.info('lcw threshold: {}'.format(args.lcw_threshold))


def main(args):
    n_entity, n_relation, data = load_dataset(os.path.join(args.data_dir, args.dataset), reciprocal=True)

    logging.info('Dataset: {}'.format(args.dataset))
    logging.info('# of entities: {}'.format(n_entity))
    logging.info('# of relations: {}'.format(n_relation))

    triple_by_relation = {key: {} for key in data.keys()}

    for split in data:
        for h, r, t in data[split]:
            if r not in triple_by_relation[split]:
                triple_by_relation[split][r] = []
            triple_by_relation[split][r].append((h, t))

    logging.info('Triples grouped by relations...')

    # Augmenting positive triples
    if args.augment:
        entity_pair_dict = dict()
        for h, r, t in data['train']:
            if (h, t) not in entity_pair_dict:
                entity_pair_dict[(h, t)] = len(entity_pair_dict)

        relation_cooccurrence = np.zeros((n_relation * 2, len(entity_pair_dict)))
        for r in triple_by_relation['train']:
            for h, t in triple_by_relation['train'][r]:
                relation_cooccurrence[r][entity_pair_dict[(h, t)]] = 1

        relation_cooccurrence = np.matmul(relation_cooccurrence, relation_cooccurrence.T)
        relation_inference = np.zeros(relation_cooccurrence.shape)

        for i in range(relation_inference.shape[0]):
            for j in range(relation_inference.shape[1]):
                relation_inference[i, j] = \
                    1.0 * relation_cooccurrence[i, j] / relation_cooccurrence[j, j]

        logging.info('Relation inference matrix constructed...')

        sub_threshold = 0.8
        logging.info('Thresholds to find subrelations: {}'.format(sub_threshold))
        for r1 in range(relation_inference.shape[0]):
            for r2 in range(relation_inference.shape[1]):
                if r1 == r2:
                    continue
                if relation_inference[r1, r2] >= sub_threshold:
                    logging.info('{} borrows triples from {}'.format(r1, r2))
                    triple_by_relation['train'][r1] += triple_by_relation['train'][r2]

            triple_by_relation['train'][r1] = list(set(triple_by_relation['train'][r1]))

        logging.info('Positive triples augmented...')

    relation_range = dict()
    for r in triple_by_relation['train']:
        for h, t in triple_by_relation['train'][r]:
            if r not in relation_range:
                relation_range[r] = []
            relation_range[r].append(t)
            relation_range[r] = list(set(relation_range[r]))

    for r in triple_by_relation['train']:
        logging.info('Relation {} range = {} / {}'.format(r, len(relation_range[r]), n_entity))

    excluded_entity = dict()
    if args.lcwa:
        lcw_index = calc_lcw_index(triple_by_relation['train'])
        for r in range(n_relation * 2):
            excluded_entity[r] = []
            if r in relation_range and lcw_index[r] >= args.lcw_threshold:
                logging.info('{} is range-constrained: {}'.format(r, lcw_index[r]))
                for i in range(n_entity):
                    if i not in set(relation_range[r]):
                        excluded_entity[r].append(i)

    record_metrics = dict()

    for r in triple_by_relation['test']:
        record_metrics[r] = {'MR': 0.0, 'MRR': 0.0, 'H@1': 0.0, 'H@3': 0.0, 'H@10': 0.0, 'n_test': 0}
        record_metrics[r]['n_test'] = len(triple_by_relation['test'][r])

    negative_sampling_space = dict()

    if args.neg_sampling == 'ont':
        head_dict = dict()
        tail_dict = dict()

        for r in relation_range:
            head_dict[r] = {relation_range[(r + n_relation) % (2 * n_relation)][i]: i
                            for i in range(len(relation_range[(r + n_relation) % (2 * n_relation)]))}
            tail_dict[r] = {relation_range[r][i]: i for i in range(len(relation_range[r]))}

        tail_cooccurrence = dict()

        for r in triple_by_relation['train']:
            tail_cooccurrence[r] = np.zeros((len(tail_dict[r]), len(head_dict[r])))
            for h, t in triple_by_relation['train'][r]:
                tail_cooccurrence[r][tail_dict[r][t]][head_dict[r][h]] = 1
            tail_cooccurrence[r] = np.matmul(tail_cooccurrence[r], tail_cooccurrence[r].T)

        logging.info('Tail co-occurrence established...')

        # Sample from low co-occurrence attributes
        for r in tail_cooccurrence:
            for i in range(tail_cooccurrence[r].shape[0]):
                negative_sampling_space[(-1, r, relation_range[r][i])] = []
                threshold = np.percentile(tail_cooccurrence[r][i], 90)
                for j in range(tail_cooccurrence[r].shape[1]):
                    if tail_cooccurrence[r][i, j] <= threshold:
                        negative_sampling_space[(-1, r, relation_range[r][i])].append(relation_range[r][j])

        del tail_cooccurrence
        gc.collect()

    filter = {'train': dict(), 'valid': dict(), 'complete': dict()}

    for split in filter:
        for r in triple_by_relation[split]:
            for h, t in triple_by_relation[split][r]:
                if (h, r) not in filter[split]:
                    filter[split][(h, r)] = set()
                filter[split][(h, r)].add(t)

    entity_embedding = np.load('{}/{}_{}_{}.npy'.format(args.emb_dir,
                                                        args.pretrained_emb,
                                                        args.dataset,
                                                        args.dim))
    relation_embedding = np.load('{}/{}_{}_{}_relation.npy'.format(args.emb_dir,
                                                                   args.pretrained_emb,
                                                                   args.dataset,
                                                                   args.dim))

    logging.info('Pretrained embedding {} loaded'.format(args.pretrained_emb))
    logging.info('Entity embedding dimension: {}'.format(entity_embedding.shape[1]))
    logging.info('Relation embedding dimension: {}'.format(relation_embedding.shape[1]))

    n_group = args.num_group
    best_kmeans = None
    lowest_inertia_ = float('inf')
    for i in range(10):
        kmeans = KMeans(n_clusters=n_group).fit(relation_embedding)
        if kmeans.inertia_ < lowest_inertia_:
            lowest_inertia_ = kmeans.inertia_
            best_kmeans = kmeans
    group_assignment = best_kmeans.predict(relation_embedding)

    group = []
    for i in range(n_group):
        group.append(np.arange(relation_embedding.shape[0])[group_assignment == i])

    group_clf = []

    for g_id in range(len(group)):
        X_train = []
        y_train = []

        for r in group[g_id]:
            logging.info("=== Current relation: {} ===".format(r))

            if r in triple_by_relation['train']:
                for h, t in triple_by_relation['train'][r]:
                    X_train.append(np.concatenate((entity_embedding[h],
                                                   relation_embedding[r],
                                                   entity_embedding[t])))
                    y_train.append(1)

                    if (-1, r, t) in negative_sampling_space:
                        negative_sample = np.random.randint(len(negative_sampling_space[(-1, r, t)]),
                                                            size=args.negative_size)
                        for i in range(args.negative_size):
                            negative_sample[i] = negative_sampling_space[(-1, r, t)][negative_sample[i]]
                    else:
                        negative_sample = np.random.randint(n_entity, size=args.negative_size)

                    mask = np.in1d(
                        negative_sample,
                        list(filter['train'][(h, r)]),
                        assume_unique=True,
                        invert=True
                    )
                    negative_sample = negative_sample[mask]

                    if len(negative_sample) == 0:
                        negative_sample = np.random.randint(n_entity, size=args.negative_size)
                        mask = np.in1d(
                            negative_sample,
                            list(filter['train'][(h, r)]),
                            assume_unique=True,
                            invert=True
                        )
                        negative_sample = negative_sample[mask]

                    for i in range(len(negative_sample)):
                        X_train.append(np.concatenate((entity_embedding[h],
                                                       relation_embedding[r],
                                                       entity_embedding[negative_sample[i]])))
                        y_train.append(0)

                    if (-1, r + n_relation, h) in negative_sampling_space:
                        negative_sample = np.random.randint(len(negative_sampling_space[(-1, r + n_relation, h)]),
                                                            size=args.negative_size)
                        for i in range(args.negative_size):
                            negative_sample[i] = negative_sampling_space[(-1, r + n_relation, h)][negative_sample[i]]
                    else:
                        negative_sample = np.random.randint(n_entity, size=args.negative_size)

                    mask = np.in1d(
                        negative_sample,
                        list(filter['train'][(t, r + n_relation)]),
                        assume_unique=True,
                        invert=True
                    )
                    negative_sample = negative_sample[mask]

                    if len(negative_sample) == 0:
                        negative_sample = np.random.randint(n_entity, size=args.negative_size)
                        mask = np.in1d(
                            negative_sample,
                            list(filter['train'][(t, r + n_relation)]),
                            assume_unique=True,
                            invert=True
                        )
                        negative_sample = negative_sample[mask]

                    for i in range(len(negative_sample)):
                        X_train.append(np.concatenate((entity_embedding[negative_sample[i]],
                                                       relation_embedding[r],
                                                       entity_embedding[t])))
                        y_train.append(0)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        logging.info("=== Dataset Constructed ===")
        logging.info("=== Start Training      ===")
        logging.info("# of training samples: {}".format(y_train.shape[0]))

        max_depth = args.max_depth
        n_estimators = args.n_estimators

        clf = xgb.XGBClassifier(objective='binary:logistic',
                                max_depth=max_depth,
                                n_estimators=n_estimators,
                                subsample=0.6,
                                eta=0.3,
                                colsample_bytree=0.6,
                                tree_method='hist',
                                min_child_weight=4,
                                use_label_encoder=False)

        eval_set = [(X_train[y_train == 1], y_train[y_train == 1]),
                    (X_train[y_train == 0], y_train[y_train == 0]),
                    (X_train, y_train)]
        clf.fit(X_train, y_train, eval_metric=['logloss'], eval_set=eval_set,
                early_stopping_rounds=50, verbose=True)

        group_clf.append(clf)

    del X_train
    del y_train
    gc.collect()

    for r in triple_by_relation['test']:
        mr = 0
        mrr = 0
        hits_1 = 0
        hits_3 = 0
        hits_10 = 0
        if r < n_relation:
            clf = group_clf[group_assignment[r]]
            for h, t in triple_by_relation['test'][r]:
                head_embedding = np.concatenate((entity_embedding[h], relation_embedding[r]))
                head_embedding = np.tile(head_embedding, (n_entity, 1))
                X_test = np.concatenate((head_embedding, entity_embedding), axis=1)

                pred = clf.predict_proba(X_test)
                score = pred[:, 1]

                if args.lcwa:
                    exclusion = list(set(excluded_entity[r] + list(filter['complete'][(h, r)])))
                else:
                    exclusion = list(filter['complete'][(h, r)])

                sorted_raw_score = np.sort(score)

                exclusion.remove(t)
                exclusion = np.array(exclusion, dtype=int)
                score[exclusion] = 0

                sorted_lcwa_score = np.sort(score)

                argsort = np.argsort(-score)

                ranking = (argsort == t).nonzero()[0][0] + 1

                mr += ranking
                mrr += 1.0 / ranking
                hits_1 += int(ranking <= 1)
                hits_3 += int(ranking <= 3)
                hits_10 += int(ranking <= 10)
        else:
            clf = group_clf[group_assignment[r - n_relation]]
            for h, t in triple_by_relation['test'][r - n_relation]:
                tail_embedding = np.concatenate((relation_embedding[r - n_relation], entity_embedding[t]))
                tail_embedding = np.tile(tail_embedding, (n_entity, 1))
                X_test = np.concatenate((entity_embedding, tail_embedding), axis=1)

                pred = clf.predict_proba(X_test)
                score = pred[:, 1]

                sorted_raw_score = np.sort(score)

                if args.lcwa:
                    exclusion = list(set(excluded_entity[r] + list(filter['complete'][(t, r)])))
                else:
                    exclusion = list(filter['complete'][(t, r)])

                exclusion.remove(h)
                exclusion = np.array(exclusion, dtype=int)
                score[exclusion] = 0

                sorted_lcwa_score = np.sort(score)

                argsort = np.argsort(-score)
                ranking = (argsort == h).nonzero()[0][0] + 1

                mr += ranking
                mrr += 1.0 / ranking
                hits_1 += int(ranking <= 1)
                hits_3 += int(ranking <= 3)
                hits_10 += int(ranking <= 10)

        record_metrics[r]['MR'] = mr / record_metrics[r]['n_test']
        record_metrics[r]['MRR'] = mrr / record_metrics[r]['n_test']
        record_metrics[r]['H@1'] = hits_1 / record_metrics[r]['n_test']
        record_metrics[r]['H@3'] = hits_3 / record_metrics[r]['n_test']
        record_metrics[r]['H@10'] = hits_10 / record_metrics[r]['n_test']

        logging.info('n_test: {}'.format(record_metrics[r]['n_test']))
        logging.info('MR: {}'.format(record_metrics[r]['MR']))
        logging.info('MRR: {}'.format(record_metrics[r]['MRR']))
        logging.info('H@1: {}'.format(record_metrics[r]['H@1']))
        logging.info('H@3: {}'.format(record_metrics[r]['H@3']))
        logging.info('H@10: {}\n'.format(record_metrics[r]['H@10']))

    total_mr = 0
    total_mrr = 0
    total_hits_1 = 0
    total_hits_3 = 0
    total_hits_10 = 0
    total_ntest = 0

    for r in record_metrics:
        total_mr += record_metrics[r]['n_test'] * record_metrics[r]['MR']
        total_mrr += record_metrics[r]['n_test'] * record_metrics[r]['MRR']
        total_hits_1 += record_metrics[r]['n_test'] * record_metrics[r]['H@1']
        total_hits_3 += record_metrics[r]['n_test'] * record_metrics[r]['H@3']
        total_hits_10 += record_metrics[r]['n_test'] * record_metrics[r]['H@10']
        total_ntest += record_metrics[r]['n_test']

    logging.info('Total number of the testing samples: {}'.format(total_ntest))

    total_mr = total_mr / total_ntest
    total_mrr = total_mrr / total_ntest
    total_hits_1 = total_hits_1 / total_ntest
    total_hits_3 = total_hits_3 / total_ntest
    total_hits_10 = total_hits_10 / total_ntest

    logging.info('MR: {}'.format(total_mr))
    logging.info('MRR: {}'.format(total_mrr))
    logging.info('H@1: {}'.format(total_hits_1))
    logging.info('H@3: {}'.format(total_hits_3))
    logging.info('H@10: {}'.format(total_hits_10))

    text_path = "{}/{}_{}_{}".format(args.output_dir, args.pretrained_emb, args.dataset, args.dim)
    text_file = os.path.join(text_path, '{}.txt'.format(args.neg_sampling))

    with open(text_file, 'w') as f:
        for r in record_metrics:
            f.write('Relation: {}\n'.format(r))
            f.write('n_test: {}\n'.format(record_metrics[r]['n_test']))
            f.write('MR: {}\n'.format(record_metrics[r]['MR']))
            f.write('MRR: {}\n'.format(record_metrics[r]['MRR']))
            f.write('H@1: {}\n'.format(record_metrics[r]['H@1']))
            f.write('H@3: {}\n'.format(record_metrics[r]['H@3']))
            f.write('H@10: {}\n\n'.format(record_metrics[r]['H@10']))


if __name__ == '__main__':
    args = parse_args()
    set_logger(args)
    main(args)
