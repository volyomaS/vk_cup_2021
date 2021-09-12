import os
import pickle
from argparse import ArgumentParser
from math import log
from typing import Dict, List, IO

from tqdm import tqdm


def calculate_recommendation(current_id: int, friends: Dict[int, List], fout: IO, retrieval_number: int = 10):
    """
    Used for calculating friends recommendations for separate user
    :param current_id: user id for which we calculating recommendations
    :param friends: dictionary with friends graph
    :param fout: IO object for writing
    :param retrieval_number: how many friends to recommend
    :return:
    """
    friends_of_current = [i[0] for i in friends[current_id]]
    friends_of_current_set = set(friends_of_current)
    friend_to_h = {x[0]: x[1] for x in friends[current_id]}

    # Our candidates for friendship are friends of friends, lets accumulate them
    candidates = []
    for friend in friends_of_current:
        candidates.extend([i[0] for i in friends[friend]])
    candidates = list(set(candidates))  # delete duplicates
    candidates.remove(current_id)  # delete current

    recommendations = [(0, 0) for i in range(len(candidates))]
    i = 0
    for candidate_id in candidates:
        # set of problem conditions
        if candidate_id <= current_id or candidate_id % 2 != 1 or candidate_id in friends_of_current_set:
            continue

        # calculating modified Adamic/Adar score as relevance
        relevance = 0
        for mutual_friend in list(friends_of_current_set.intersection(set([i[0] for i in friends[candidate_id]]))):
            h1, h2 = friend_to_h.setdefault(mutual_friend, 0), 0

            for k in friends[candidate_id]:
                if k[0] == mutual_friend:
                    h2 = k[1]
                    break

            if h1 == 0:
                h1 = 1e-2
            if h2 == 0:
                h2 = 1e-2

            relevance += h1 * h2 / log(len(friends[mutual_friend]))

        recommendations[i] = (candidate_id, relevance)
        i += 1

    recommendations.sort(key=lambda x: x[1], reverse=True)

    recommendations_ids = [x[0] for x in recommendations[:retrieval_number] if x != (0, 0)]
    fout.write(f"{current_id}: {','.join(map(str, recommendations_ids))}\n")


def inference(input_path: str, output_path: str):
    """
    Used for inference
    :param input_path: path to directory with friends.pkl and baseline.txt files
    :param output_path: path to directory where to save ans.txt file
    :return:
    """
    with open(os.path.join(input_path, "friends.pkl"), "rb") as fin:
        friends = pickle.load(fin)

    # syntactic sugar
    test = []
    with open(os.path.join(input_path, "baseline.txt"), "r") as fin:
        for line in fin:
            t = line.split(":")[0]
            test.append(int(t))
        test.sort()

    fout = open(os.path.join(output_path, "ans.txt"), "w")
    for t in tqdm(test):
        calculate_recommendation(t, friends, fout)


def calculate_friends(input_path: str, output_path: str):
    """
    Used for calculation friends graph and dumping resulting dictionary to file
    :param input_path: path to directory with train.csv file
    :param output_path: path to directory where to save output pkl file
    :return:
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    friends = {}
    with open(os.path.join(input_path, "train.csv"), "r") as fin:
        next(fin)
        for line in tqdm(fin):
            id1, id2, t, h = map(int, line.split(","))

            if id1 not in friends:
                friends[id1] = [(id2, h)]
            else:
                friends[id1].append((id2, h))
            if id2 not in friends:
                friends[id2] = [(id1, h)]
            else:
                friends[id2].append((id1, h))

    with open(os.path.join(output_path, "friends.pkl"), "wb") as fin:
        pickle.dump(friends, fin)


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-t", "--task", choices=["calculate_friends", "inference"], required=True)
    arg_parser.add_argument("-i", "--input", required=True, help="Path to input directory")
    arg_parser.add_argument("-o", "--output", required=True, help="Path to output directory")
    return arg_parser


if __name__ == "__main__":
    arg_parser = configure_arg_parser()
    args = arg_parser.parse_args()
    if args.task == "calculate_friends":
        calculate_friends(input_path=args.input, output_path=args.output)
    elif args.task == "inference":
        inference(input_path=args.input, output_path=args.output)
