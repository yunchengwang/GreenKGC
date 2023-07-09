import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import load_dataset


def find_best_partition(f_1d, y, bins=32, metric='entropy'):
    best_error = float('inf')
    f_min, f_max = np.min(f_1d), np.max(f_1d)
    bin_width = (f_max - f_min) / bins
    for i in range(1, bins):
        partition_point = f_min + i * bin_width
        y_l, y_r = y[f_1d <= partition_point], y[f_1d > partition_point]
        partition_error = get_partition_error(y_l, y_r, metric)
        print(i, partition_error)
        if partition_error < best_error:
            best_error = partition_error
    return best_error


def get_partition_error(y_l, y_r, metric):
    n1, n2 = len(y_l), len(y_r)
    if metric == 'entropy':
        lp = y_l.mean()
        if lp == 1 or lp == 0:
            lh = 0.0
        else:
            lh = np.sum(-y_l * np.log2(lp) - (1 - y_l) * np.log2(1 - lp))
        rp = y_r.mean()
        if rp == 1 or rp == 0:
            rh = 0.0
        else:
            rh = np.sum(-y_r * np.log2(rp) - (1 - y_r) * np.log2(1 - rp))
        return (lh + rh) / (n1 + n2)
    elif metric == 'accuracy':
        l_p1 = y_l.mean()
        r_p1 = y_r.mean()

        if l_p1 > r_p1:
            lacc = l_p1
            racc = 1 - r_p1
        else:
            lacc = 1 - l_p1
            racc = r_p1
        return 1.0 - (n1 * lacc + n2 * racc) / (n1 + n2)
    else:
        print('Unsupported error')
        return 0


def main():
    data_dir = 'data'
    dataset = sys.argv[1]

    n_entity, n_relation, data = load_dataset(os.path.join(data_dir, dataset))

    valid_candidate = dict()
    for h, r, t in data['train']:
        if (h, r) not in valid_candidate:
            valid_candidate[(h, r)] = set()
        valid_candidate[(h, r)].add(t)

        if (t, r + n_relation) not in valid_candidate:
            valid_candidate[(t, r + n_relation)] = set()
        valid_candidate[(t, r + n_relation)].add(h)

    feature_dir = 'features'
    feature = sys.argv[2]
    dim = int(sys.argv[3])

    entity_embedding = np.load(os.path.join(feature_dir, '{}_{}_{}.npy'.format(feature, dataset, dim)))
    relation_embedding = np.load(os.path.join(feature_dir, '{}_{}_{}_relation.npy'.format(feature, dataset, dim)))

    if feature == 'RotatE':
        relation_embedding = np.concatenate([np.cos(relation_embedding), np.sin(relation_embedding)], axis=-1)

    entity_embedding = entity_embedding.reshape((n_entity, -1, dim))
    relation_embedding = relation_embedding.reshape((n_relation, -1, dim))

    fig_dir = 'figures'

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    dft_error = dict()

    for i in range(dim):
        print('Dimension {}'.format(i))
        f_train = []
        y_train = []
        for h, r, t in data['train']:
            f_h = entity_embedding[h, :, i]
            f_r = relation_embedding[r, :, i]
            f_t = entity_embedding[t, :, i]
            f_train.append(np.concatenate([f_h, f_r, f_t], axis=-1))
            y_train.append(1)

            ct = np.random.randint(n_entity)

            if ct not in valid_candidate[(h, r)]:
                f_h = entity_embedding[h, :, i]
                f_r = relation_embedding[r, :, i]
                f_t = entity_embedding[ct, :, i]
                f_train.append(np.concatenate([f_h, f_r, f_t], axis=-1))
                y_train.append(0)

            ch = np.random.randint(n_entity)

            if ct not in valid_candidate[(t, r + n_relation)]:
                f_h = entity_embedding[ch, :, i]
                f_r = relation_embedding[r, :, i]
                f_t = entity_embedding[t, :, i]
                f_train.append(np.concatenate([f_h, f_r, f_t], axis=-1))
                y_train.append(0)

        f_train = np.array(f_train)
        y_train = np.array(y_train)
        clf = LogisticRegression(fit_intercept=False)
        clf.fit(f_train, y_train)
        f_1d = np.matmul(f_train, clf.coef_.T)
        dft_error[i] = find_best_partition(f_1d.reshape(-1), y_train, bins=32, metric='entropy')

    sorted_dft_error = {k: v for k, v in sorted(dft_error.items(), key=lambda item: item[1])}

    for new_dim in [32, 100]:
        selected_feat = np.array(list(sorted_dft_error.keys()))[:new_dim]
        pruned_entity_embedding = entity_embedding[:, :, selected_feat].reshape(n_entity, -1)
        pruned_relation_embedding = relation_embedding[:, :, selected_feat].reshape(n_relation, -1)

        np.save(
            os.path.join(feature_dir, '{}_pruned_{}_{}.npy'.format(feature, dataset, new_dim)),
            pruned_entity_embedding
        )
        np.save(
            os.path.join(feature_dir, '{}_pruned_{}_{}_relation.npy'.format(feature, dataset, new_dim)),
            pruned_relation_embedding
        )


if __name__ == '__main__':
    main()
