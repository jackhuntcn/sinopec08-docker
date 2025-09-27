import warnings
warnings.simplefilter('ignore')
import argparse
import os
import gc
import time
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD, SparsePCA
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
from scipy import linalg
import scipy.sparse as sp
from scipy.special import iv
from scipy.sparse import csr_matrix
import networkx as nx
from gensim.models import Word2Vec
import jieba


class ProNE():
    def __init__(self, G, emb_size=128, step=10, theta=0.5, mu=0.2, n_iter=5, random_state=2025):
        self.G = G
        self.emb_size = emb_size
        self.G = self.G.to_undirected()
        self.node_number = self.G.number_of_nodes()
        self.random_state = random_state
        self.step = step
        self.theta = theta
        self.mu = mu
        self.n_iter = n_iter
        mat = scipy.sparse.lil_matrix((self.node_number, self.node_number))
        for e in tqdm(self.G.edges()):
            if e[0] != e[1]:
                mat[int(e[0]), int(e[1])] = 1
                mat[int(e[1]), int(e[0])] = 1
        self.mat = scipy.sparse.csr_matrix(mat)

    def get_embedding_rand(self, matrix):
        # Sparse randomized tSVD for fast embedding
        t1 = time.time()
        l = matrix.shape[0]
        smat = scipy.sparse.csc_matrix(matrix)  # convert to sparse CSC format
        U, Sigma, VT = randomized_svd(smat, n_components=self.emb_size, n_iter=self.n_iter, random_state=self.random_state)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        return U

    def get_embedding_dense(self, matrix, emb_size):
        # get dense embedding via SVD
        t1 = time.time()
        U, s, Vh = linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
        U = np.array(U)
        U = U[:, :emb_size]
        s = s[:emb_size]
        s = np.sqrt(s)
        U = U * s
        U = preprocessing.normalize(U, "l2")
        return U

    def fit(self, tran, mask):
        # Network Embedding as Sparse Matrix Factorization
        t1 = time.time()
        l1 = 0.75
        C1 = preprocessing.normalize(tran, "l1")
        neg = np.array(C1.sum(axis=0))[0] ** l1
        neg = neg / neg.sum()
        neg = scipy.sparse.diags(neg, format="csr")
        neg = mask.dot(neg)
        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1
        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)
        C1 -= neg
        F = C1
        features_matrix = self.get_embedding_rand(F)
        return features_matrix

    def chebyshev_gaussian(self, A, a, order=10, mu=0.5, s=0.5):
        t1 = time.time()
        if order == 1:
            return a
        A = sp.eye(self.node_number) + A
        DA = preprocessing.normalize(A, norm='l1')
        L = sp.eye(self.node_number) - DA
        M = L - mu * sp.eye(self.node_number)
        Lx0 = a
        Lx1 = M.dot(a)
        Lx1 = 0.5 * M.dot(Lx1) - a
        conv = iv(0, s) * Lx0
        conv -= 2 * iv(1, s) * Lx1
        for i in range(2, order):
            Lx2 = M.dot(Lx1)
            Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
            if i % 2 == 0:
                conv += 2 * iv(i, s) * Lx2
            else:
                conv -= 2 * iv(i, s) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
            del Lx2
        mm = A.dot(a - conv)
        self.embeddings = self.get_embedding_dense(mm, self.emb_size)
        return self.embeddings

    def transform(self):
        if self.embeddings is None:
            return {}
        self.embeddings = pd.DataFrame(self.embeddings)
        self.embeddings.columns = ['ProNE_Emb_{}'.format(i) for i in range(len(self.embeddings.columns))]
        self.embeddings = self.embeddings.reset_index().rename(columns={'index' : 'nodes'}).sort_values(by=['nodes'],ascending=True).reset_index(drop=True)
        return self.embeddings


def display(df):
    print(df)


def gen_seq_feats(input_path):

    train = pl.read_csv(f'{input_path}/cust_wallet_detail_train.csv')
    train = train.with_columns(
        pl.col('sale_time').str.strptime(pl.Datetime, '%Y/%m/%d %H:%M').alias('sale_time')
    )
    train = train.sort('sale_time', descending=False)
    train = train.with_columns(
        pl.when(pl.col('coupon_code').is_null())
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias('label')
    ).with_columns(
        pl.lit('train').alias('data_type')
    )

    coupon_train = pl.read_csv(f'{input_path}/cust_coupon_detail_send_train.csv')
    coupon_train = coupon_train.with_columns(
        pl.col('voucherstarttime').cast(pl.String).str.strptime(pl.Datetime, '%Y%m%d').alias('voucherstarttime'),
        pl.col('voucherendtime').cast(pl.String).str.strptime(pl.Datetime, '%Y%m%d').alias('voucherendtime')
    ).unique()
    coupon_train = coupon_train.sort('voucherstarttime', descending=False)

    test = pl.read_csv(f'{input_path}/cust_wallet_detail_validation_without_truth.csv')
    test = test.with_columns(
        pl.col('sale_time').str.strptime(pl.Datetime, '%Y/%m/%d %H:%M').alias('sale_time')
    )
    test = test.sort('sale_time', descending=False)
    test = test.with_columns(
        pl.lit('test').alias('data_type'),
        pl.lit(-1).cast(pl.Float64).alias('coupon_amt'),
        pl.lit('').alias('coupon_code'),
        pl.lit(-1).cast(pl.Int32).alias('label')
    )
    test = test.select(train.columns)

    coupon_test = pl.read_csv(f'{input_path}/cust_coupon_detail_send_validation.csv')
    coupon_test = coupon_test.with_columns(
        pl.col('voucherstarttime').cast(pl.String).str.strptime(pl.Datetime, '%Y%m%d').alias('voucherstarttime'),
        pl.col('voucherendtime').cast(pl.String).str.strptime(pl.Datetime, '%Y%m%d').alias('voucherendtime')
    ).unique()
    coupon_test = coupon_test.sort('voucherstarttime', descending=False)

    data = pl.concat([train, test])
    data = data.sort('sale_time', descending=False)
    data = data.with_columns(
        pl.Series('row_id', list(range(len(data))))
    )

    data = data.with_columns(
        pl.col('sale_time').dt.date().alias('sale_time_date'),
        (pl.col('tran_amt').cast(pl.Utf8) + " + " + pl.col('discounts_amt').cast(pl.Utf8)).alias('trans_and_discounts'),
    )

    coupon_data = pl.concat([coupon_train, coupon_test])
    coupon_data = coupon_data.sort('voucherstarttime', descending=False)
    coupon_data = coupon_data.with_columns(
        pl.Series('row_id', list(range(len(coupon_data))))
    )

    coupon_data = coupon_data.with_columns(
        pl.col('voucherstarttime').dt.date().alias('voucherstarttime_date'),
        pl.col('voucherendtime').dt.date().alias('voucherendtime_date'),
        (pl.col('voucherendtime') - pl.col('voucherstarttime')).dt.total_days().alias('voucher_days_gap'),
    )

    # 清理异常用户
    data = data.filter(
        ~pl.col('membercode').is_in([1010005382088, 1033005085671, 1034010552989])
    )
    coupon_data = coupon_data.filter(
        ~pl.col('membercode').is_in([1010005382088, 1033005085671, 1034010552989])
    )

    # tran_amt_membercode_tfidf
    df = data.group_by('membercode').agg(
        pl.col('tran_amt').alias('tran_amt_list')
    )

    texts = list()
    for item in df['tran_amt_list'].to_list():
        texts.append(' '.join([str(w) for w in item]))

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",
                                 ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)

    n_components = 32
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    df_emb = pl.DataFrame(reduced_matrix)
    df_emb.columns = [f'tran_amt_tfidf_{i}' for i in range(n_components)]
    df_emb = df_emb.with_columns(
        pl.Series('membercode', df['membercode'].to_list()),
    )
    display(df_emb)
    os.makedirs('feats', exist_ok=True)
    df_emb.write_parquet('feats/tran_amt_membercode_tfidf.parquet')

    # trans_and_discounts_membercode_tfidf
    df = data.group_by('trans_and_discounts').agg(
        pl.col('membercode').alias('membercode_list')
    )

    texts = list()
    for item in df['membercode_list'].to_list():
        texts.append(' '.join([str(w) for w in item]))

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",
                                 ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)

    n_components = 16
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    df_emb = pl.DataFrame(reduced_matrix)
    df_emb.columns = [f'trans_and_discounts_membercode_tfidf_{i}' for i in range(n_components)]
    df_emb = df_emb.with_columns(
        pl.Series('trans_and_discounts', df['trans_and_discounts'].to_list()),
    )
    display(df_emb)
    os.makedirs('feats', exist_ok=True)
    df_emb.write_parquet('feats/trans_and_discounts_membercode_tfidf.parquet')

    # station_code_membercode_tfidf
    df = data.group_by('station_code').agg(
        pl.col('membercode').alias('membercode_list')
    )

    texts = list()
    for item in df['membercode_list'].to_list():
        texts.append(' '.join([str(w) for w in item]))

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",
                                 ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)

    n_components = 16
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    df_emb = pl.DataFrame(reduced_matrix)
    df_emb.columns = [f'station_code_membercode_tfidf_{i}' for i in range(n_components)]
    df_emb = df_emb.with_columns(
        pl.Series('station_code', df['station_code'].to_list()),
    )
    display(df_emb)
    os.makedirs('feats', exist_ok=True)
    df_emb.write_parquet('feats/station_code_membercode_tfidf.parquet')


    # trans_and_discounts_prone
    i = 'membercode'
    j = 'trans_and_discounts'

    data1 = data.select([i, j]).to_pandas()
    did_lbl, vid_lbl = LabelEncoder(), LabelEncoder()
    data1[f'new_{i}'] = did_lbl.fit_transform(data1[i])
    data1[f'new_{j}'] = vid_lbl.fit_transform(data1[j])
    data1[f'new_{j}'] += data1[f'new_{i}'].max() + 1

    emb_size = 16
    G = nx.Graph()
    G.add_edges_from(data1[[f'new_{i}', f'new_{j}']].values)
    model = ProNE(G, emb_size=emb_size, n_iter=10, step=50, random_state=2025)
    features_matrix = model.fit(model.mat, model.mat)
    model.chebyshev_gaussian(model.mat, features_matrix, model.step, model.mu, model.theta)
    emb = model.transform()
    emb_did = emb[emb['nodes'].isin(data1[f'new_{i}'])]
    emb_did['nodes'] = did_lbl.inverse_transform(emb_did['nodes'])
    emb_did.rename(columns={'nodes' : i}, inplace=True)
    emb_vid = emb[emb['nodes'].isin(data1['new_%s'%(j)])]
    emb_vid['nodes'] = vid_lbl.inverse_transform(emb_vid['nodes'] -  data1[f'new_{i}'].max() - 1)
    emb_vid.rename(columns={'nodes' : j}, inplace=True)

    emb_did.columns = ['membercode'] + [f'trans_and_discounts_ProNE_Emb_{i}' for i in range(emb_size)]
    emb_did = emb_did.reset_index(drop=True)
    display(emb_did)

    emb_did.to_parquet(f'feats/trans_and_discounts_prone.parquet')


    # transactionorgcode_prone
    i = 'membercode'
    j = 'transactionorgcode'

    data1 = data.select([i, j]).to_pandas()
    did_lbl, vid_lbl = LabelEncoder(), LabelEncoder()
    data1[f'new_{i}'] = did_lbl.fit_transform(data1[i])
    data1[f'new_{j}'] = vid_lbl.fit_transform(data1[j])
    data1[f'new_{j}'] += data1[f'new_{i}'].max() + 1

    emb_size = 16
    G = nx.Graph()
    G.add_edges_from(data1[[f'new_{i}', f'new_{j}']].values)
    model = ProNE(G, emb_size=emb_size, n_iter=10, step=50, random_state=2025)
    features_matrix = model.fit(model.mat, model.mat)
    model.chebyshev_gaussian(model.mat, features_matrix, model.step, model.mu, model.theta)
    emb = model.transform()
    emb_did = emb[emb['nodes'].isin(data1[f'new_{i}'])]
    emb_did['nodes'] = did_lbl.inverse_transform(emb_did['nodes'])
    emb_did.rename(columns={'nodes' : i}, inplace=True)
    emb_vid = emb[emb['nodes'].isin(data1['new_%s'%(j)])]
    emb_vid['nodes'] = vid_lbl.inverse_transform(emb_vid['nodes'] -  data1[f'new_{i}'].max() - 1)
    emb_vid.rename(columns={'nodes' : j}, inplace=True)

    emb_did.columns = ['membercode'] + [f'transactionorgcode_ProNE_Emb_{i}' for i in range(emb_size)]
    emb_did = emb_did.reset_index(drop=True)
    display(emb_did)

    emb_did.to_parquet(f'feats/transactionorgcode_prone.parquet')


    # station_code_prone
    i = 'membercode'
    j = 'station_code'

    data1 = data.select([i, j]).to_pandas()
    did_lbl, vid_lbl = LabelEncoder(), LabelEncoder()
    data1[f'new_{i}'] = did_lbl.fit_transform(data1[i])
    data1[f'new_{j}'] = vid_lbl.fit_transform(data1[j])
    data1[f'new_{j}'] += data1[f'new_{i}'].max() + 1

    emb_size = 16
    G = nx.Graph()
    G.add_edges_from(data1[[f'new_{i}', f'new_{j}']].values)
    model = ProNE(G, emb_size=emb_size, n_iter=10, step=50, random_state=2025)
    features_matrix = model.fit(model.mat, model.mat)
    model.chebyshev_gaussian(model.mat, features_matrix, model.step, model.mu, model.theta)
    emb = model.transform()
    emb_did = emb[emb['nodes'].isin(data1[f'new_{i}'])]
    emb_did['nodes'] = did_lbl.inverse_transform(emb_did['nodes'])
    emb_did.rename(columns={'nodes' : i}, inplace=True)
    emb_vid = emb[emb['nodes'].isin(data1['new_%s'%(j)])]
    emb_vid['nodes'] = vid_lbl.inverse_transform(emb_vid['nodes'] -  data1[f'new_{i}'].max() - 1)
    emb_vid.rename(columns={'nodes' : j}, inplace=True)

    emb_did.columns = ['membercode'] + [f'station_code_ProNE_Emb_{i}' for i in range(emb_size)]
    emb_did = emb_did.reset_index(drop=True)
    display(emb_did)

    emb_did.to_parquet(f'feats/station_code_prone.parquet')


    # trans_and_discounts_membercode_prone
    i = 'trans_and_discounts'
    j = 'membercode'

    data1 = data.select([i, j]).to_pandas()
    did_lbl, vid_lbl = LabelEncoder(), LabelEncoder()
    data1[f'new_{i}'] = did_lbl.fit_transform(data1[i])
    data1[f'new_{j}'] = vid_lbl.fit_transform(data1[j])
    data1[f'new_{j}'] += data1[f'new_{i}'].max() + 1

    emb_size = 16
    G = nx.Graph()
    G.add_edges_from(data1[[f'new_{i}', f'new_{j}']].values)
    model = ProNE(G, emb_size=emb_size, n_iter=10, step=50, random_state=2025)
    features_matrix = model.fit(model.mat, model.mat)
    model.chebyshev_gaussian(model.mat, features_matrix, model.step, model.mu, model.theta)
    emb = model.transform()
    emb_did = emb[emb['nodes'].isin(data1[f'new_{i}'])]
    emb_did['nodes'] = did_lbl.inverse_transform(emb_did['nodes'])
    emb_did.rename(columns={'nodes' : i}, inplace=True)
    emb_vid = emb[emb['nodes'].isin(data1['new_%s'%(j)])]
    emb_vid['nodes'] = vid_lbl.inverse_transform(emb_vid['nodes'] -  data1[f'new_{i}'].max() - 1)
    emb_vid.rename(columns={'nodes' : j}, inplace=True)

    emb_did.columns = ['trans_and_discounts'] + [f'trans_and_discounts_membercode_ProNE_Emb_{i}' for i in range(emb_size)]
    emb_did = emb_did.reset_index(drop=True)
    display(emb_did)

    emb_did.to_parquet(f'feats/trans_and_discounts_membercode_prone.parquet')

    # tran_amt_w2v
    df = data.group_by('membercode').agg(
        pl.col('tran_amt').alias('tran_amt_list')
    )

    texts = list()
    for item in df['tran_amt_list'].to_list():
        texts.append(' '.join([str(w) for w in item]))

    vector_size = 8
    window = 20
    min_count = 1
    workers = 1

    model = Word2Vec(
        sentences=texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        epochs=5
    )

    embeddings = []
    for tokens in texts:
        if tokens:
            token_vectors = [model.wv[token] for token in tokens if token in model.wv]
            if token_vectors:
                doc_vector = np.mean(token_vectors, axis=0)
            else:
                doc_vector = np.zeros(vector_size)
        else:
            doc_vector = np.zeros(vector_size)
        embeddings.append(doc_vector)

    df_emb = pl.DataFrame(np.array(embeddings))
    df_emb.columns = [f'tran_amt_w2v_{i}' for i in range(vector_size)]
    df_emb = df_emb.with_columns(
        pl.Series('membercode', df['membercode'].to_list()),
    )
    display(df_emb)
    os.makedirs('feats', exist_ok=True)
    df_emb.write_parquet('feats/tran_amt_w2v.parquet')

    # voucherrulename_w2v
    df = coupon_data.group_by('membercode').agg(
        pl.col('voucherrulename').alias('voucherrulename_list')
    )

    texts = list()
    for item in df['voucherrulename_list'].to_list():
        texts.append(' '.join(jieba.cut(' '.join(item))))

    vector_size = 8
    window = 10
    min_count = 1
    workers = 1

    model = Word2Vec(
        sentences=texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        epochs=5
    )

    embeddings = []
    for tokens in texts:
        if tokens:
            token_vectors = [model.wv[token] for token in tokens if token in model.wv]
            if token_vectors:
                doc_vector = np.mean(token_vectors, axis=0)
            else:
                doc_vector = np.zeros(vector_size)
        else:
            doc_vector = np.zeros(vector_size)
        embeddings.append(doc_vector)

    df_emb = pl.DataFrame(np.array(embeddings))
    df_emb.columns = [f'voucherrulename_w2v_{i}' for i in range(vector_size)]
    df_emb = df_emb.with_columns(
        pl.Series('membercode', df['membercode'].to_list()),
    )
    display(df_emb)
    os.makedirs('feats', exist_ok=True)
    df_emb.write_parquet('feats/voucherrulename_w2v.parquet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    args = parser.parse_args()
    gen_seq_feats(args.input_path)
