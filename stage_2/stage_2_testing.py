from typing import List

import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler


def calculate_probable_age(usersEducationFeatures):
    prob_age = {}
    grads_count = {}
    age_diff1 = 17  # age difference for school
    age_diff2 = 22  # age difference for university
    for index in usersEducationFeatures.index:
        count = 0
        skip = False

        if not pd.isnull(usersEducationFeatures.at[index, "school_education"]):
            prob_age[usersEducationFeatures.at[index, "uid"]] = (
                2021 + age_diff1 - usersEducationFeatures.at[index, "school_education"]
            )
            skip = True
        for i in range(1, 8):
            if skip:
                break
            if not pd.isnull(usersEducationFeatures.at[index, f"graduation_{i}"]):
                prob_age[usersEducationFeatures.at[index, "uid"]] = (
                    2021 + age_diff2 - usersEducationFeatures.at[index, f"graduation_{i}"]
                )
                skip = True

        if not pd.isnull(usersEducationFeatures.at[index, "school_education"]):
            count += 1
        for i in range(1, 8):
            if not pd.isnull(usersEducationFeatures.at[index, f"graduation_{i}"]):
                count += 1

        grads_count[usersEducationFeatures.at[index, "uid"]] = count
    return prob_age, grads_count


def get_prob_age(uids, prob_age) -> List[int]:
    res = [0] * len(uids)
    for i, uid in enumerate(uids):
        res[i] = prob_age.setdefault(uid, 0)
    return res


def get_grads_count(uids, grads_count) -> List[int]:
    res = [0] * len(uids)
    for i, uid in enumerate(uids):
        res[i] = grads_count.setdefault(uid, 0)
    return res


def get_groups_count(uids, usersGroups):
    tmp = usersGroups.groupby("uid").count()
    groups_count = [0] * len(uids)
    for i, uid in enumerate(uids):
        try:
            groups_count[i] = tmp.at[uid, "gid"]
        except:
            continue
    return groups_count


def get_mean_and_median_group(uids, gid2age, uid_groups):
    mean_group = [0.0] * len(uids)
    median_group = [0.0] * len(uids)
    for i, uid in enumerate(uids):
        try:
            tmp = [gid2age[x] for x in uid_groups[uid]]
            mean_group[i] = sum(tmp) / len(tmp)
            median_group[i] = np.median(tmp)
        except:
            continue
    return mean_group, median_group


def get_mean_and_median_friends(uids, uid2age, uid_friends):
    mean_friends = [0.0] * len(uids)
    median_friends = [0.0] * len(uids)
    mean_friends2 = [0.0] * len(uids)
    for i, uid in enumerate(uids):
        try:
            tmp = []
            if uid in uid_friends and len(uid_friends[uid]) < 42:
                for friend in uid_friends[uid]:
                    if friend in uid_friends:
                        for f2 in uid_friends[friend]:
                            if f2 != uid and f2 in uid2age:
                                tmp.append(uid2age[f2])
            mean_friends2[i] = sum(tmp) / len(tmp) if len(tmp) != 0 else 0
            tmp = [uid2age[x] for x in uid_friends[uid] if x in uid2age]
            mean_friends[i] = sum(tmp) / len(tmp) if len(tmp) != 0 else 0.0
            median_friends[i] = np.median(tmp) if len(tmp) != 0 else 0.0
        except:
            continue
    return mean_friends, median_friends, mean_friends2


def main():
    with open("gid2age.pkl", "rb") as fin:
        gid2age = pickle.load(fin)
    with open("uid2age.pkl", "rb") as fin:
        uid2age = pickle.load(fin)
    with open("uid_friends.pkl", "rb") as fin:
        uid_friends = pickle.load(fin)
    with open("scaler.pkl", "rb") as fin:
        scaler = pickle.load(fin)
    model = CatBoostRegressor()
    model.load_model("model")

    test = pd.read_csv("/tmp/data/test.csv")
    testEducationFeatures = pd.read_csv("/tmp/data/testEducationFeatures.csv")
    testGroups = pd.read_csv("/tmp/data/testGroups.csv")

    test["cfriends"] = 0
    for index in test.index:
        uid = test.at[index, "uid"]
        if uid in uid_friends:
            test.at[index, "cfriends"] = len(uid_friends[uid])
        else:
            test.at[index, "cfriends"] = 0

    prob_age, grads_count = calculate_probable_age(testEducationFeatures)
    test["prob_age"] = get_prob_age(test.uid, prob_age)
    test["grads_count"] = get_grads_count(test.uid, grads_count)

    test["groups_count"] = get_groups_count(test.uid, testGroups)

    uid_groups = {}
    for index in testGroups.index:
        uid = testGroups.at[index, "uid"]
        uid_groups[uid] = uid_groups.setdefault(uid, []) + [testGroups.at[index, "gid"]]

    test["mean_group_age"], test["median_group_age"] = get_mean_and_median_group(test.uid, gid2age, uid_groups)

    test["mean_friends_age"], test["median_friends_age"], test["mean_friends2_age"] = get_mean_and_median_friends(
        test.uid, uid2age, uid_friends
    )

    test["is_prob_age"] = test.prob_age != 0
    test["is_group_age"] = test.mean_group_age != 0
    test["is_friends_age"] = test.mean_friends_age != 0

    X_test = scaler.transform(test.drop(["uid"], axis=1))

    y_pred = model.predict(X_test)

    res = pd.DataFrame({"uid": test.uid, "age": y_pred})

    res.to_csv("/var/log/result", header=True, index=False)


if __name__ == "__main__":
    main()
