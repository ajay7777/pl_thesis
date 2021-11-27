import pandas as pd
import re

import torch

def augment_train_phone_language_pairs(data):

    data_de_pairs = data[data.lang_1 == "de"]
    data2 = data[['cluster_id_2',
       'offer_id_2', 'category_2', 'subcategory_2', 'lang_2', 'title_2',
       'description_2', 'ean_2', 'mpn_2']]
    data1 = data[['cluster_id_1', 'offer_id_1', 'category_1', 'subcategory_1', 'lang_1',
       'title_1', 'description_1', 'ean_1', 'mpn_1']]
    data2.columns = ['cluster_id_1', 'offer_id_1', 'category_1', 'subcategory_1', 'lang_1',
       'title_1', 'description_1', 'ean_1', 'mpn_1']
    all_unique_offers = pd.concat([data1, data2]).drop_duplicates().reset_index(drop=True)
    all_unique_offers_en = all_unique_offers[all_unique_offers.lang_1=='en']

    d = []
    for row in data_de_pairs.itertuples(index=True, name='Pandas'):
        row_new = all_unique_offers_en[all_unique_offers_en.cluster_id_1 == row.cluster_id_1].sample().iloc[0]
        d.append(
            {
                'cluster_id_1': row_new["cluster_id_1"],
                'offer_id_1': row_new.offer_id_1,
                'category_1':  row_new.category_1,
                'subcategory_1':row_new.subcategory_1,
                'lang_1':row_new.lang_1,
                'title_1':row_new.title_1,
                'description_1':row_new.description_1,
                'ean_1':row_new.ean_1,
                'mpn_1':row_new.mpn_1,
                'cluster_id_2':row.cluster_id_2,
                'offer_id_2':row.offer_id_2,
                'category_2':row.category_2,
                'subcategory_2':row.subcategory_2,
                'lang_2':row.lang_2,
                'title_2':row.title_2,
                'description_2':row.description_2,
                'ean_2':row.ean_2,
                'mpn_2':row.mpn_2,
                'label':row.label,
                'hardness':row.hardness
            }
        )
    return pd.concat([data, pd.DataFrame(d)])

def prep_data_pair(
        train_data,
        test_data,
        use_description):
    """
    Runs simple preprocessing and encoding of test and training data
    :param train_data:
    :param test_data:
    :return: training_set, test_set
    """

    train_data["content_1"] = train_data["title_1"]
    train_data["content_2"] = train_data["title_2"]
    test_data["content_1"] = test_data["title_1"]
    test_data["content_2"] = test_data["title_2"]

    if use_description:
        train_data.loc[train_data.description_1.notna(), "content_1"] = train_data.loc[
                                                                            train_data.description_1.notna(), "title_1"] + ' ' + \
                                                                        train_data.loc[
                                                                            train_data.description_1.notna(), "description_1"]
        train_data.loc[train_data.description_2.notna(), "content_2"] = train_data.loc[
                                                                            train_data.description_2.notna(), "title_2"] + ' ' + \
                                                                        train_data.loc[
                                                                            train_data.description_2.notna(), "description_2"]
        test_data.loc[test_data.description_1.notna(), "content_1"] = test_data.loc[
                                                                          test_data.description_1.notna(), "title_1"] + ' ' + \
                                                                      test_data.loc[
                                                                          test_data.description_1.notna(), "description_1"]
        test_data.loc[test_data.description_2.notna(), "content_2"] = test_data.loc[
                                                                          test_data.description_2.notna(), "title_2"] + ' ' + \
                                                                      test_data.loc[
                                                                          test_data.description_2.notna(), "description_2"]

    # remove MPN and EAN from titles
    train_data["content_1"] = train_data.apply(
        lambda row: remove_identifier_from_content_pair(row, "category_1", "content_1", "ean_1"), axis=1)
    train_data["content_2"] = train_data.apply(
        lambda row: remove_identifier_from_content_pair(row, "category_2", "content_2", "ean_2"), axis=1)

    test_data["content_1"] = test_data.apply(
        lambda row: remove_identifier_from_content_pair(row, "category_1", "content_1", "ean_1"), axis=1)
    test_data["content_2"] = test_data.apply(
        lambda row: remove_identifier_from_content_pair(row, "category_2", "content_2", "ean_2"), axis=1)

    return train_data, test_data

def remove_identifier_from_content_pair(row, category, content, ean):
    """
    For offers, that contain MPN or EAN in their title/description, remove it
    :return:
    """

    # remove MPN and EAN (use different logic for toy and phone)
    row[content] = re.sub(re.escape(str(row[ean])), '', row[content])

    if row[category] == 'toy':
        row[content] = re.sub(r"\d{5}|\d{4}", '', row[content])
    elif row[category] == 'phone':
        row[content] = re.sub(r"\bM[A-Z]{1,3}\d*[A-Z]*\d{1,2}[A-Z]{1,2}[A-Z]|SM.{0,1}[A-Z][\d]{3}[A-Z]{1,2}|GA\d{5}",
                              '',
                              row[content])
    return row[content]

class CustomDataset(torch.utils.data.Dataset):
    """
    Custom implementation of abstract torch.utils.data.Dataset class
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)