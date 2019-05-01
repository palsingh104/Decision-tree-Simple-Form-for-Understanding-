import math
from collections import Counter, defaultdict
from functools import partial

inputs = [
({'Country':'USA', 'PrimaryLanguage':'Java', 'Research':'no', 'phd':'no'}, False),
({'Country':'USA', 'PrimaryLanguage':'Java', 'Research':'no', 'phd':'yes'}, False),
({'Country':'India', 'PrimaryLanguage':'Python', 'Research':'no', 'phd':'no'}, True),
({'Country':'China', 'PrimaryLanguage':'Python', 'Research':'no', 'phd':'no'}, True),
({'Country':'China', 'PrimaryLanguage':'R', 'Research':'yes', 'phd':'no'}, True),
({'Country':'China', 'PrimaryLanguage':'R', 'Research':'yes', 'phd':'yes'}, False),
({'Country':'India', 'PrimaryLanguage':'R', 'Research':'yes', 'phd':'yes'}, True),
({'Country':'USA', 'PrimaryLanguage':'Python', 'Research':'no', 'phd':'no'}, False),
({'Country':'USA', 'PrimaryLanguage':'R', 'Research':'yes', 'phd':'no'}, True),
({'Country':'China', 'PrimaryLanguage':'Python', 'Research':'yes', 'phd':'no'}, True),
({'Country':'USA', 'PrimaryLanguage':'Python', 'Research':'yes', 'phd':'yes'}, True),
({'Country':'India', 'PrimaryLanguage':'Python', 'Research':'no', 'phd':'yes'}, True),
({'Country':'India', 'PrimaryLanguage':'Java', 'Research':'yes', 'phd':'no'}, True),
({'Country':'China', 'PrimaryLanguage':'Python', 'Research':'no', 'phd':'yes'}, False)
]

def entropy(class_probabilities):

    return sum(-p * math.log(p, 2)
    for p in class_probabilities
    if p) # ignore zero probabilities

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_entropy(subsets):
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count
               for subset in subsets)



def partition_by(inputs, attribute):

    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute] # get the value of the specified attribute
        groups[key].append(input) # then add this input to the correct list
    return groups

def partition_entropy_by(inputs, attribute):

    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

for key in ['Country','PrimaryLanguage','Research','phd']:
    print (key, partition_entropy_by(inputs, key))

print("#########################")

USA_inputs = [(input, label)
for input, label in inputs if input["Country"] == "USA"]
for key in ['PrimaryLanguage', 'Research', 'phd']:
    print (key, partition_entropy_by(USA_inputs, key))



def classify(tree, input):

    if tree in [True, False]:
        return tree

    attribute, subtree_dict = tree
    subtree_key = input.get(attribute) # None if input is missing attribute
    if subtree_key not in subtree_dict: # if no subtree for key,
        subtree_key = None # we'll use the None subtree
    subtree = subtree_dict[subtree_key] # choose the appropriate subtree
    return classify(subtree, input) # and use it to classify the input

def build_tree_id3(inputs, split_candidates=None):
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()
    # count Trues and Falses in the inputs
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues
    if num_trues == 0: return False  # no Trues? return a "False" leaf
    if num_falses == 0: return True  # no Falses? return a "True" leaf
    if not split_candidates:  # if no split candidates left
        return num_trues >= num_falses  # return the majority leaf
    # otherwise, split on the best attribute
    best_attribute = min(split_candidates,
                         key=partial(partition_entropy_by, inputs))
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates
                      if a != best_attribute]
    # recursively build the subtrees
    subtrees = {attribute_value: build_tree_id3(subset, new_candidates)
                for attribute_value, subset in partitions.items()}
    subtrees[None] = num_trues > num_falses  # default case
    return (best_attribute, subtrees)

print("#########################")

tree = build_tree_id3(inputs)

print(classify(tree, { "Country" : "China",
"PrimaryLanguage" : "Java",
"Research" : "yes",
"phd" : "no"} ) )

print(classify(tree, { "Country" : "China",
"PrimaryLanguage" : "Java",
"Research" : "yes",
"phd" : "yes"} ) )

print("#########################")

print(classify(tree, { "Country" : "Intern" } ) )
print(classify(tree, { "Country" : "USA" } ) )

print(classify(tree, { "Country" : "India",
"PrimaryLanguage" : "Java",
"Research" : "yes",
"phd" : "no"} ) )

