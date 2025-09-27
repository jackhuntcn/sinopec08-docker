import warnings
warnings.simplefilter('ignore')
import argparse
import os
import gc
import numpy as np
import polars as pl
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

def train_and_predict(input_path, output_path):
    # 训练集交易表
    train = pl.read_csv(f'{input_path}/cust_wallet_detail_train.csv')
    # 预处理时间类型
    train = train.with_columns(
        pl.col('sale_time').str.strptime(pl.Datetime, '%Y/%m/%d %H:%M').alias('sale_time')
    )
    train = train.sort('sale_time', descending=False)
    # 构建标签, coupon_code 非 null 为 1
    train = train.with_columns(
        pl.when(pl.col('coupon_code').is_null())
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias('label')
    ).with_columns(
        pl.lit('train').alias('data_type')
    )

    # 训练集发放表
    coupon_train = pl.read_csv(f'{input_path}/cust_coupon_detail_send_train.csv')
    # 预处理时间类型
    coupon_train = coupon_train.with_columns(
        pl.col('voucherstarttime').cast(pl.String).str.strptime(pl.Datetime, '%Y%m%d').alias('voucherstarttime'),
        pl.col('voucherendtime').cast(pl.String).str.strptime(pl.Datetime, '%Y%m%d').alias('voucherendtime')
    ).unique()
    coupon_train = coupon_train.sort('voucherstarttime', descending=False)

    # 测试集交易表
    test = pl.read_csv(f'{input_path}/cust_wallet_detail_validation_without_truth.csv')
    # 预处理时间类型
    test = test.with_columns(
        pl.col('sale_time').str.strptime(pl.Datetime, '%Y/%m/%d %H:%M').alias('sale_time')
    )
    test = test.sort('sale_time', descending=False)
    # 为了合并, dummy 创建一些只在训练集里有的列
    test = test.with_columns(
        pl.lit('test').alias('data_type'),
        pl.lit(-1).cast(pl.Float64).alias('coupon_amt'),
        pl.lit('').alias('coupon_code'),
        pl.lit(-1).cast(pl.Int32).alias('label')
    )
    test = test.select(train.columns)

    # 测试集发放表
    coupon_test = pl.read_csv(f'{input_path}/cust_coupon_detail_send_validation.csv')
    # 预处理时间类型
    coupon_test = coupon_test.with_columns(
        pl.col('voucherstarttime').cast(pl.String).str.strptime(pl.Datetime, '%Y%m%d').alias('voucherstarttime'),
        pl.col('voucherendtime').cast(pl.String).str.strptime(pl.Datetime, '%Y%m%d').alias('voucherendtime')
    ).unique()
    coupon_test = coupon_test.sort('voucherstarttime', descending=False)

    # 合并数据并重新按时间排序
    wallet_data = pl.concat([train, test])
    wallet_data = wallet_data.sort('sale_time', descending=False)
    wallet_data = wallet_data.with_columns(
        pl.Series('row_id', list(range(len(wallet_data))))
    )
    coupon_data = pl.concat([coupon_train, coupon_test])
    coupon_data = coupon_data.sort('voucherstarttime', descending=False)
    coupon_data = coupon_data.with_columns(
        pl.Series('row_id', list(range(len(coupon_data))))
    )

    # 清理训练集中的异常用户
    wallet_data = wallet_data.filter(
        ~pl.col('membercode').is_in([1010005382088, 1033005085671, 1034010552989])
    )
    coupon_data = coupon_data.filter(
        ~pl.col('membercode').is_in([1010005382088, 1033005085671, 1034010552989])
    )

    def make_feats(wallet_data, coupon_data, embs1, embs2, embs3):
        # 发放表 member 维度特征
        member_coupon_feats = coupon_data.with_columns(
            pl.col('topamount').fill_null(-100000).alias('topamount'),
        ).with_columns(
            pl.col('voucherstarttime').dt.month().alias('voucherstarttime_month'),
            pl.col('voucherendtime').dt.month().alias('voucherendtime_month'),
            (pl.col('voucherendtime') - pl.col('voucherstarttime')).dt.total_days()
            .alias('voucher_days_gap'),
            (pl.col('cashvalue') / pl.col('topamount')).alias('discount_percentile'),
        ).group_by('membercode').agg(
            pl.col('row_id').count().alias('member_coupon_count'),
            *[pl.col(col).n_unique().alias(f'member_{col}_nunique')
              for col in ['marketcode', 'marketrulenumber', 'marketprovince',
                          'voucherrulecode', 'voucherrulename', 'vouchertype',
                          'fulltype', 'usechannel', 'voucherstarttime_month',
                          'voucherendtime_month']],
            *[pl.col(col).min().alias(f'member_{col}_min')
              for col in ['voucherstarttime', 'voucherendtime', 'cashvalue', 'topamount',
                          'endnumber', 'voucher_days_gap', 'discount_percentile']],
            *[pl.col(col).max().alias(f'member_{col}_max')
              for col in ['voucherstarttime', 'voucherendtime', 'cashvalue', 'topamount',
                          'endnumber', 'voucher_days_gap', 'discount_percentile']],
            *[pl.col(col).mean().alias(f'member_{col}_mean')
              for col in ['cashvalue', 'topamount', 'endnumber',
                          'voucher_days_gap', 'discount_percentile']],
            *[pl.col(col).std().alias(f'member_{col}_std')
              for col in ['cashvalue', 'topamount', 'endnumber',
                          'voucher_days_gap', 'discount_percentile']],
        )
        # 发放表 province 维度特征 (transactionorgcode / attributionorgcode)
        province_coupon_feats = coupon_data.with_columns(
            (pl.col('voucherendtime') - pl.col('voucherstarttime')).dt.total_days()
            .alias('voucher_days_gap'),
        ).group_by('marketprovince').agg(
            pl.col('row_id').count()
                .alias('province_coupon_count'),
            *[pl.col(col).n_unique().alias(f'province_{col}_nunique')
              for col in ['marketcode', 'membercode']],
            *[pl.col(col).min().alias(f'province_{col}_min')
              for col in ['voucherstarttime', 'voucherendtime', 'cashvalue',
                          'topamount', 'voucher_days_gap']],
            *[pl.col(col).max().alias(f'province_{col}_max')
              for col in ['voucherstarttime', 'voucherendtime', 'cashvalue',
                          'topamount', 'voucher_days_gap']],
            *[pl.col(col).mean().alias(f'province_{col}_mean')
              for col in ['cashvalue', 'topamount', 'voucher_days_gap']],
            *[pl.col(col).std().alias(f'province_{col}_std')
              for col in ['cashvalue', 'topamount', 'voucher_days_gap']],
        )
        register_coupon_feats = province_coupon_feats.clone()
        register_coupon_feats.columns = ['attributionorgcode_filled'] +\
                [c.replace('province_', 'register_') for c in province_coupon_feats.columns[1:]]
        # attributionorgcode 填充
        df = wallet_data.filter(
            pl.col('attributionorgcode').is_not_null()
        ).unique('membercode').select(['membercode', 'attributionorgcode'])
        di = {row['membercode']: row['attributionorgcode'] for row in df.to_dicts()}
        wallet_data = wallet_data.with_columns(
            pl.col('attributionorgcode').replace(di).cast(pl.Int64).alias('attributionorgcode_filled')
        )
        # 合并特征
        df_feats = wallet_data.join(
            member_coupon_feats, on='membercode', how='left'
        ).join(
            province_coupon_feats, left_on='transactionorgcode', right_on='marketprovince', how='left'
        ).join(
            register_coupon_feats, on='attributionorgcode_filled', how='left'
        )
        # 交易表特征
        df_feats = df_feats.with_columns(
            # 手工类别特征
            pl.col('sale_time').dt.date().alias('sale_time_date'),
            (pl.col('tran_amt').cast(pl.Utf8) + " + " + pl.col('discounts_amt').cast(pl.Utf8))
            .alias('trans_and_discounts'),
            pl.col('user_id').shift(1).over('membercode').alias('user_id_shift1'),
            pl.col('user_id').shift(-1).over('membercode').alias('user_id_shift-1'),
            # 数值特征
            (pl.col('discounts_amt') / (pl.col('tran_amt'))).alias('discounts_tran_ratio'),
            *[(pl.col('tran_amt') / pl.col(f'member_topamount_{m}'))
              .alias(f'tran_amt_topamount_{m}_ratio') for m in ['mean', 'max', 'min']],
            # 原始时间特征
            pl.col('sale_time').dt.year().alias('sale_time_year'),
            pl.col('sale_time').dt.month().alias('sale_time_month'),
            pl.col('sale_time').dt.day().alias('sale_time_day'),
            pl.col('sale_time').dt.hour().alias('sale_time_hour'),
            pl.col('sale_time').dt.weekday().alias('sale_time_weekday'),
        ).with_columns(
            # member 维度统计特征
            *[pl.col(col).max().over('membercode').alias(f'member_{col}_max')
              for col in ['tran_amt', 'discounts_amt']],
            *[pl.col(col).min().over('membercode').alias(f'member_{col}_min')
              for col in ['tran_amt', 'discounts_amt']],
            *[pl.col(col).mean().over('membercode').alias(f'member_{col}_mean')
              for col in ['tran_amt', 'discounts_amt']],
            *[pl.col(col).std().over('membercode').alias(f'member_{col}_std')
              for col in ['tran_amt', 'discounts_amt']],
            pl.col('order_no').count().over('membercode').alias('member_order_count'),
            *[pl.col(col).n_unique().over('membercode').alias(f'member_{col}_nunique')
              for col in ['station_code', 'tran_amt', 'discounts_amt',
                          'transactionorgcode', 'sale_time_date']],
            pl.col('sale_time_date').n_unique().over(['membercode', 'station_code'])
            .alias('member_station_date_nunique'),
            pl.col('order_no').count().over(['membercode', 'sale_time_date'])
            .alias('member_sale_time_date_count'),
            pl.col('sale_time_weekday').n_unique().over(['membercode', 'station_code'])
            .alias('member_station_weekday_nunique'),
            # 其他维度统计特征
            *[pl.col('order_no').count().over(col).alias(f'{col}_count')
              for col in ['station_code', 'sale_time_date', 'tran_amt', 'discounts_amt',
                          'trans_and_discounts', 'transactionorgcode', 'attributionorgcode']],
            # 时间比较特征
            *[(pl.col(col1) - pl.col(col2)).dt.total_days().alias(f'{col1}_{col2}_days_gap')
              for col1, col2 in [('member_voucherendtime_max', 'member_voucherstarttime_min'),
                                 ('province_voucherendtime_max', 'province_voucherstarttime_min'),
                                 ('register_voucherendtime_max', 'register_voucherstarttime_min'),
                                 ('sale_time', 'member_voucherstarttime_min'),
                                 ('member_voucherendtime_max', 'sale_time'),
                                 ('sale_time', 'member_voucherstarttime_max'),
                                 ('member_voucherendtime_min', 'sale_time'),
                                 ('sale_time', 'province_voucherstarttime_min'),
                                 ('province_voucherendtime_max', 'sale_time'),
                                 ('sale_time', 'province_voucherstarttime_max'),
                                 ('province_voucherendtime_min', 'sale_time'),
                                 ('sale_time', 'register_voucherstarttime_min'),
                                 ('register_voucherendtime_max', 'sale_time'),
                                 ('sale_time', 'register_voucherstarttime_max'),
                                 ('register_voucherendtime_min', 'sale_time')]],
        ).with_columns(
            # 变化类特征
            ## sale_time
            *[pl.col('sale_time').diff(i).over('membercode').dt.total_seconds()
              .fill_null(-100000).alias(f'member_sale_time_diff{i}') for i in range(1, 6)],
            *[pl.col('sale_time').diff(-i).over('membercode').dt.total_seconds()
              .fill_null(-100000).alias(f'member_sale_time_diff-{i}') for i in range(1, 6)],
            ## station_code
            *[pl.col('station_code').diff(i).over('membercode')
              .alias(f'member_station_diff{i}') for i in range(1, 4)],
            ## transactionorgcode
            *[pl.col('transactionorgcode').diff(i).over('membercode')
              .alias(f'member_transactionorgcode_diff{i}') for i in range(1, 4)],
            *[pl.col('transactionorgcode').diff(-i).over('membercode')
              .alias(f'member_transactionorgcode_diff-{i}') for i in range(1, 4)],
            ## row_id / tran_amt / discounts_tran_ratio
            ## member_sale_time_date_count / tran_amt_count
            ## transactionorgcode_count / attributionorgcode_count
            *[pl.col(col).pct_change(i).over('membercode')
              .alias(f'member_{col}_pct_diff{i}') for i in range(1, 4) for col in[
                  'row_id', 'tran_amt', 'discounts_tran_ratio',
                  'member_sale_time_date_count', 'tran_amt_count',
                  'transactionorgcode_count', 'attributionorgcode_count'
              ]],
            *[pl.col(col).pct_change(-i).over('membercode')
              .alias(f'member_{col}_pct_diff-{i}') for i in range(1, 4) for col in[
                  'row_id', 'tran_amt', 'discounts_tran_ratio',
                  'member_sale_time_date_count', 'tran_amt_count',
                  'transactionorgcode_count', 'attributionorgcode_count'
              ]],
        ).with_columns(
            # 比较特征
            (pl.col('attributionorgcode') == pl.col('transactionorgcode'))
            .cast(pl.Int32).fill_null(-1).alias('is_same_place'),
            (pl.col('user_id_shift1') == pl.col('user_id'))
            .cast(pl.Int32).alias('same_user_id_diff1'),
            (pl.col('user_id_shift-1') == pl.col('user_id'))
            .cast(pl.Int32).alias('same_user_id_diff-1'),
        ).with_columns(
            # member 截面排序特征
            *[(pl.col(col).rank(descending=True, method='ordinal') / pl.col(col).count())
              .over('membercode').alias(f'{col}_rank_over_member')
              for col in [
                  'row_id', 'tran_amt', 'tran_amt_count',
                  'discounts_tran_ratio', 'trans_and_discounts_count',
                  'transactionorgcode_count', 'station_code_count',
                  'member_sale_time_date_count', 'member_sale_time_date_nunique',
                  'member_station_date_nunique', 'member_station_weekday_nunique',
                  'province_cashvalue_min', 'province_cashvalue_max',
                  'province_topamount_min', 'province_topamount_max',
                  'sale_time_member_voucherstarttime_min_days_gap',
                  'sale_time_member_voucherstarttime_max_days_gap']],
        ).with_columns(
            # member, date 截面排序特征
            *[(pl.col(col).rank(descending=True, method='ordinal') / pl.col(col).count())
              .over(['membercode', 'sale_time_date']).alias(f'{col}_rank_over_member_date')
              for col in ['tran_amt', 'trans_and_discounts_count']],
        ).with_columns(
            # 处理 inf 值, 否则 XGBoost 会报错
            *[pl.when(pl.col(f'member_{col}_pct_diff{i}').is_infinite())
              .then(pl.lit(0)).otherwise(pl.col(f'member_{col}_pct_diff{i}'))
              .alias(f'member_{col}_pct_diff{i}') for i in range(1, 4) for col in[
                  'row_id', 'tran_amt', 'discounts_tran_ratio',
                  'member_sale_time_date_count', 'tran_amt_count',
                  'transactionorgcode_count', 'attributionorgcode_count'
              ]],
            *[pl.when(pl.col(f'member_{col}_pct_diff-{i}').is_infinite())
              .then(pl.lit(0)).otherwise(pl.col(f'member_{col}_pct_diff-{i}'))
              .alias(f'member_{col}_pct_diff-{i}') for i in range(1, 4) for col in[
                  'row_id', 'tran_amt', 'discounts_tran_ratio',
                  'member_sale_time_date_count', 'tran_amt_count',
                  'transactionorgcode_count', 'attributionorgcode_count'
              ]],
        ).with_columns(
            # 一些奇怪的特征
            pl.col('order_no').str.slice(-4).cast(pl.Int64).alias('order_no_last4str'),
            pl.col('order_no').str.replace_all(r'[0-9]', '').alias('order_no_type'),
            pl.col('external_order_no').str.slice(0,3).alias('external_order_no_type'),
        ).with_columns(
            pl.col('order_no_type').cast(pl.Categorical).to_physical()
            .alias('order_no_type'),
            pl.col('external_order_no_type').cast(pl.Categorical).to_physical()
            .alias('external_order_no_type'),
        ).with_columns(
            pl.col('order_no_type').n_unique().over('membercode')
            .alias('member_order_no_type_nunique'),
            pl.col('external_order_no_type').n_unique().over('membercode')
            .alias('member_external_order_no_type_nunique'),
        )

        for emb in embs1:
            df_feats = df_feats.join(emb, on='membercode', how='left')

        for emb in embs2:
            df_feats = df_feats.join(emb, on='trans_and_discounts', how='left')

        for emb in embs3:
            df_feats = df_feats.join(emb, on='station_code', how='left')

        return df_feats


    embs1 = []
    for parq_file in [
        'feats/tran_amt_membercode_tfidf.parquet',
        'feats/tran_amt_w2v.parquet',
        'feats/trans_and_discounts_prone.parquet',
        'feats/transactionorgcode_prone.parquet',
        'feats/station_code_prone.parquet',
        'feats/voucherrulename_w2v.parquet',
    ]:
        embs1.append(pl.read_parquet(parq_file))

    embs2 = []
    for parq_file in [
        'feats/trans_and_discounts_membercode_prone.parquet',
        'feats/trans_and_discounts_membercode_tfidf.parquet',
    ]:
        embs2.append(pl.read_parquet(parq_file))

    embs3 = []
    for parq_file in [
        'feats/station_code_membercode_tfidf.parquet',
    ]:
        embs3.append(pl.read_parquet(parq_file))

    df_data = make_feats(wallet_data, coupon_data, embs1, embs2, embs3)

    print(df_data.head(20))

    df_train = df_data.filter(pl.col('data_type') == 'train')
    df_test = df_data.filter(pl.col('data_type') == 'test')

    useless_cols = ['order_no', 'external_order_no', 'row_id',
                    'user_id', 'membercode', 'coupon_amt',
                    'station_name', 'sale_time', 'sale_time_date', 'trans_and_discounts',
                    'coupon_code', 'label', 'data_type', 'user_id_shift1', 'user_id_shift-1',
                    'attributionorgcode_filled'] +\
                   ['receivable_amt', 'point_amt']
    features = [c for c in df_train.columns if c not in useless_cols]
    print(len(features))

    X = df_train.select(features).to_numpy()
    y = df_train['label'].to_numpy()
    g = df_train['membercode'].to_numpy()
    X_test = df_test.select(features).to_numpy()
    print(X.shape, y.shape, X_test.shape)

    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)
    sub_sample = pl.read_csv(f'{input_path}/sample_submission.csv', columns=['id']).unique()

    lgb_params = {
        'boosting_type': 'dart',
        'drop_rate': 0.1,
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': 12,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)

    oof_preds = np.zeros(len(X))
    models = []
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(X, y, g)):
        print(f"正在训练第 {fold + 1} 折...")
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=5000,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=True),
                lgb.log_evaluation(period=1000),
            ]
        )
        valid_preds = model.predict(X_valid, num_iteration=model.best_iteration)
        oof_preds[valid_idx] = valid_preds
        fold_auc = roc_auc_score(y_valid, valid_preds)
        fold_scores.append(fold_auc)
        print(f"第 {fold + 1} 折 AUC: {fold_auc:.6f}")
        models.append(model)

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n五折交叉验证完成！")
    print(f"各折AUC分数: {[f'{score:.6f}' for score in fold_scores]}")
    print(f"平均AUC: {np.mean(fold_scores):.6f}")
    print(f"OOF AUC: {overall_auc:.6f}")

    X_test = df_test.select(features).to_numpy()
    pred = models[0].predict(X_test, num_iteration=models[0].best_iteration) / len(models)
    for model in models[1:]:
        pred += model.predict(X_test, num_iteration=model.best_iteration) / len(models)
    df_sub = df_test.select(['order_no']).with_columns(
        pl.Series('predict', pred)
    )
    df_sub = sub_sample.join(df_sub, left_on='id', right_on='order_no', how='left')
    print(df_sub)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'submission.csv')
    df_sub.write_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the prediction results')
    args = parser.parse_args()
    train_and_predict(args.input_path, args.output_path)
